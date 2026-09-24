#include "branchscore/backend_context.hpp"
#include "branchscore/decision.hpp"
#include "branchscore/gemma4_decision_engine.hpp"
#include "branchscore/json.hpp"
#include "branchscore/model_loader.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = branchscore::json::Value;
using Object = Json::Object;
using Array = Json::Array;
using Clock = std::chrono::steady_clock;

struct InputRow {
    std::string id;
    std::size_t line = 0;
    branchscore::DecisionRequest request;
};

Json number(const double value) {
    if (!std::isfinite(value)) throw std::runtime_error("benchmark value is not finite");
    return Json(value);
}

Json size_number(const std::size_t value) {
    return number(static_cast<double>(value));
}

const Json & required(const Json & object, const std::string & key) {
    const auto * value = object.find(key);
    if (value == nullptr) throw std::runtime_error("JSON object is missing '" + key + "'");
    return *value;
}

const std::string & required_string(const Json & object, const std::string & key) {
    return required(object, key).string();
}

std::optional<std::string> optional_string(const Json & object, const std::string & key) {
    const auto * value = object.find(key);
    if (value == nullptr) return std::nullopt;
    return value->string();
}

std::vector<InputRow> read_rows(
    const std::filesystem::path & input_path) {
    std::ifstream input(input_path);
    if (!input) throw std::runtime_error("failed to open input JSONL '" + input_path.string() + "'");

    std::vector<InputRow> rows;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;

        const auto value = branchscore::json::parse(line);
        const auto id = required_string(value, "id");
        if (id.empty()) throw std::runtime_error("input line " + std::to_string(line_number) +
                                                 " has an empty id");

        branchscore::DecisionRequest request;
        request.state = required_string(value, "state");
        request.question = required_string(value, "question");
        const auto & options = required(value, "options").array();
        request.options.reserve(options.size());
        for (const auto & option : options) {
            request.options.push_back(branchscore::DecisionOption{
                required_string(option, "id"),
                required_string(option, "description"),
            });
        }
        if (const auto image = optional_string(value, "image")) {
            if (image->empty()) throw std::runtime_error("input line " +
                std::to_string(line_number) + " has an empty image path");
            const auto image_path = std::filesystem::path(*image);
            request.image_path = image_path.is_absolute()
                ? image_path.lexically_normal().string()
                : (input_path.parent_path() / image_path).lexically_normal().string();
        }
        if (const auto template_file = optional_string(value, "chat_template_file")) {
            request.chat_template_file = *template_file;
        }
        rows.push_back(InputRow{id, line_number, std::move(request)});
    }
    if (input.bad()) throw std::runtime_error("failed while reading input JSONL");
    if (rows.empty()) throw std::runtime_error("input JSONL contains no request rows");
    return rows;
}

Json timing_json(const branchscore::TimingInfo & timing) {
    Object result;
    result.emplace("prompt_rendering_ms", number(timing.prompt_rendering_ms));
    result.emplace("tokenization_ms", number(timing.tokenization_ms));
    result.emplace("image_preprocessing_ms", number(timing.image_preprocessing_ms));
    result.emplace("vision_ms", number(timing.vision_ms));
    result.emplace("vision_backend_copy_ms", number(timing.vision_backend_copy_ms));
    result.emplace("vision_synchronization_ms", number(timing.vision_synchronization_ms));
    result.emplace("vision_graph_node_count", size_number(timing.vision_graph_node_count));
    result.emplace("vision_attention_path", timing.vision_attention_path);
    result.emplace("prefill_ms", number(timing.prefill_ms));
    result.emplace("prefill_backend_copy_ms", number(timing.prefill_backend_copy_ms));
    result.emplace("prefill_synchronization_ms", number(timing.prefill_synchronization_ms));
    result.emplace("prefill_graph_node_count", size_number(timing.prefill_graph_node_count));
    result.emplace("readout_ms", number(timing.readout_ms));
    result.emplace("readout_backend_copy_ms", number(timing.readout_backend_copy_ms));
    result.emplace("readout_synchronization_ms", number(timing.readout_synchronization_ms));
    result.emplace("readout_graph_node_count", size_number(timing.readout_graph_node_count));
    result.emplace("normalization_ms", number(timing.normalization_ms));
    result.emplace("request_total_ms", number(timing.request_total_ms));
    return Json(std::move(result));
}

Json prompt_json(const branchscore::DecisionResult & result) {
    Object prompt;
    prompt.emplace("renderer_id", result.prompt_format.renderer_id);
    prompt.emplace("model_family", result.prompt_format.model_family);
    prompt.emplace("effective_source", result.prompt_format.effective_source);
    prompt.emplace(
        "reasoning_policy",
        branchscore::reasoning_policy_name(result.prompt_format.prompt_policy.reasoning));
    prompt.emplace("identity", result.rendered_prompt_identity);
    prompt.emplace("prompt_token_count", size_number(result.rendered_prompt_token_ids.size()));
    Array prefix_ids;
    for (const auto id : result.rendered_prompt_token_ids) prefix_ids.emplace_back(number(id));
    prompt.emplace("prompt_token_ids", Json(std::move(prefix_ids)));
    prompt.emplace("gguf_chat_template_present",
                   Json(result.prompt_format.gguf_chat_template_present));
    prompt.emplace("gguf_chat_template_used",
                   Json(result.prompt_format.gguf_chat_template_used));
    if (result.prompt_format.requested_template_file) {
        prompt.emplace("requested_template_file", *result.prompt_format.requested_template_file);
        prompt.emplace("requested_template_applied",
                       Json(result.prompt_format.requested_template_applied));
    }
    return Json(std::move(prompt));
}

Json row_json(
    const InputRow & input,
    const branchscore::DecisionResult & result,
    const double measured_request_ms) {
    Object row;
    row.emplace("kind", "decision");
    row.emplace("schema_version", size_number(result.schema_version));
    row.emplace("request_id", input.id);
    row.emplace("input_line", size_number(input.line));
    row.emplace("state_byte_count", size_number(input.request.state.size()));
    row.emplace("question", input.request.question);
    if (input.request.image_path) row.emplace("image_path", *input.request.image_path);

    Array options;
    for (const auto & score : result.option_scores) {
        const auto & input_option = input.request.options.at(score.input_index);
        Object option;
        option.emplace("id", score.option_id);
        option.emplace("input_index", size_number(score.input_index));
        option.emplace("description", input_option.description);
        option.emplace("answer_label", score.answer_label);
        option.emplace("answer_token_id", number(score.answer_token_id));
        option.emplace("raw_score", number(score.raw_score));
        option.emplace("relative_probability", number(score.relative_probability));
        options.emplace_back(Json(std::move(option)));
    }
    row.emplace("options", Json(std::move(options)));
    row.emplace("selected_id", result.selected_id);
    row.emplace("selected_index", size_number(result.selected_index));
    row.emplace("exact_tie", Json(result.exact_tie));
    row.emplace("scoring_basis", result.scoring_basis);
    row.emplace("readout_id", result.readout_id);
    row.emplace("terminator_scored", Json(result.terminator_scored));
    row.emplace("prompt", prompt_json(result));
    row.emplace("timings_ms", timing_json(result.timings));
    row.emplace("measured_request_ms", number(measured_request_ms));
    return Json(std::move(row));
}

Json run_json(
    const std::filesystem::path & input_path,
    const std::filesystem::path & model_path,
    const std::filesystem::path & mmproj_path,
    const branchscore::BackendContext & backend,
    const branchscore::ModelBundle & model,
    const std::size_t warmup_count,
    const std::size_t row_count) {
    const auto & text = model.text_config();
    const auto & vision = model.vision_config();
    Object run;
    run.emplace("kind", "run");
    run.emplace("schema_version", size_number(2));
    run.emplace("renderer_id", "gemma4-categorical-v1");
    run.emplace("readout_id", "gemma4-next-token-categorical-v1");
    run.emplace("scoring_basis", "answer_slot_logit");
    run.emplace("input_path", input_path.string());
    run.emplace("model_path", model_path.string());
    run.emplace("mmproj_path", mmproj_path.string());
    run.emplace("backend_name", backend.device().name);
    run.emplace("backend_family", backend.device().family);
    run.emplace("backend_type", backend.device().type);
    run.emplace("model_architecture", text.architecture);
    run.emplace("model_name", text.name);
    run.emplace("context_length", size_number(text.context_length));
    run.emplace("embedding_length", size_number(text.embedding_length));
    run.emplace("block_count", size_number(text.block_count));
    run.emplace("vocabulary_size", size_number(text.vocabulary_size));
    run.emplace("text_tensor_count", size_number(model.text_tensor_count()));
    run.emplace("text_weight_bytes", size_number(model.text_weight_bytes()));
    run.emplace("vision_projector", vision.projector_type);
    run.emplace("vision_tensor_count", size_number(model.vision_tensor_count()));
    run.emplace("vision_weight_bytes", size_number(model.vision_weight_bytes()));
    run.emplace("warmup_count", size_number(warmup_count));
    run.emplace("row_count", size_number(row_count));
    run.emplace("timing_boundary",
                "measured request interval is one Gemma4DecisionEngine::evaluate call; "
                "model loading, warmup, and JSONL writes are excluded");
    return Json(std::move(run));
}

double percentile(std::vector<double> values, const double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto position = fraction * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(position);
    const auto upper = std::min(lower + 1, values.size() - 1);
    const auto weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

Json aggregate_json(
    const std::vector<double> & measured_ms,
    const std::vector<branchscore::TimingInfo> & timings,
    const double wall_ms) {
    if (measured_ms.empty() || timings.size() != measured_ms.size()) {
        throw std::runtime_error("cannot aggregate an empty benchmark");
    }
    auto average = [&](const auto getter) {
        double total = 0.0;
        for (const auto & timing : timings) total += getter(timing);
        return total / static_cast<double>(timings.size());
    };

    Object timing;
    timing.emplace("prompt_rendering_ms_mean", number(average([](const auto & v) {
        return v.prompt_rendering_ms;
    })));
    timing.emplace("tokenization_ms_mean", number(average([](const auto & v) {
        return v.tokenization_ms;
    })));
    timing.emplace("image_preprocessing_ms_mean", number(average([](const auto & v) {
        return v.image_preprocessing_ms;
    })));
    timing.emplace("vision_ms_mean", number(average([](const auto & v) {
        return v.vision_ms;
    })));
    timing.emplace("vision_backend_copy_ms_mean", number(average([](const auto & v) {
        return v.vision_backend_copy_ms;
    })));
    timing.emplace("vision_synchronization_ms_mean", number(average([](const auto & v) {
        return v.vision_synchronization_ms;
    })));
    timing.emplace("vision_graph_node_count_mean", number(average([](const auto & v) {
        return static_cast<double>(v.vision_graph_node_count);
    })));
    timing.emplace("prefill_ms_mean", number(average([](const auto & v) {
        return v.prefill_ms;
    })));
    timing.emplace("prefill_backend_copy_ms_mean", number(average([](const auto & v) {
        return v.prefill_backend_copy_ms;
    })));
    timing.emplace("prefill_synchronization_ms_mean", number(average([](const auto & v) {
        return v.prefill_synchronization_ms;
    })));
    timing.emplace("prefill_graph_node_count_mean", number(average([](const auto & v) {
        return static_cast<double>(v.prefill_graph_node_count);
    })));
    timing.emplace("readout_ms_mean", number(average([](const auto & v) {
        return v.readout_ms;
    })));
    timing.emplace("readout_backend_copy_ms_mean", number(average([](const auto & v) {
        return v.readout_backend_copy_ms;
    })));
    timing.emplace("readout_synchronization_ms_mean", number(average([](const auto & v) {
        return v.readout_synchronization_ms;
    })));
    timing.emplace("readout_graph_node_count_mean", number(average([](const auto & v) {
        return static_cast<double>(v.readout_graph_node_count);
    })));
    timing.emplace("normalization_ms_mean", number(average([](const auto & v) {
        return v.normalization_ms;
    })));
    timing.emplace("request_total_ms_mean", number(average([](const auto & v) {
        return v.request_total_ms;
    })));

    Object result;
    result.emplace("kind", "aggregate");
    result.emplace("schema_version", size_number(2));
    result.emplace("row_count", size_number(measured_ms.size()));
    result.emplace("measured_wall_ms", number(wall_ms));
    result.emplace("decisions_per_second",
                   number(1000.0 * static_cast<double>(measured_ms.size()) / wall_ms));
    result.emplace("request_latency_p50_ms", number(percentile(measured_ms, 0.50)));
    result.emplace("request_latency_p95_ms", number(percentile(measured_ms, 0.95)));
    result.emplace("timing_means_ms", Json(std::move(timing)));
    result.emplace("timing_boundary",
                   "wall time covers sequential evaluate calls only; model loading, warmup, "
                   "and JSONL writes are excluded");
    return Json(std::move(result));
}

void print_help() {
    std::cout
        << "Usage: branchscore-bench --model FILE --mmproj FILE --input FILE --output FILE\n"
        << "                        [--backend NAME] [--warmup COUNT]\n"
        << "Input rows require id, state, question, and options[{id,description}].\n"
        << "Optional image paths are resolved relative to the input JSONL file.\n"
        << "The output must not already exist; output is run, decision, and aggregate JSONL.\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        std::string selector = "auto";
        std::filesystem::path model_path;
        std::filesystem::path mmproj_path;
        std::filesystem::path input_path;
        std::filesystem::path output_path;
        std::size_t warmup_count = 1;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto require_argument = [&](const char * name) -> std::string {
                if (i + 1 >= argc) throw std::runtime_error(std::string(name) +
                                                             " requires a value");
                return argv[++i];
            };
            if (arg == "--model") model_path = require_argument("--model");
            else if (arg == "--mmproj") mmproj_path = require_argument("--mmproj");
            else if (arg == "--input") input_path = require_argument("--input");
            else if (arg == "--output") output_path = require_argument("--output");
            else if (arg == "--backend") selector = require_argument("--backend");
            else if (arg == "--warmup") {
                const auto value = require_argument("--warmup");
                std::size_t consumed = 0;
                warmup_count = std::stoull(value, &consumed);
                if (consumed != value.size()) {
                    throw std::runtime_error("--warmup expects an integer");
                }
            } else if (arg == "--help") {
                print_help();
                return 0;
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }

        if (model_path.empty() || mmproj_path.empty() || input_path.empty() || output_path.empty()) {
            throw std::runtime_error("--model, --mmproj, --input, and --output are required");
        }
        if (std::filesystem::exists(output_path)) {
            throw std::runtime_error("refusing to overwrite existing output '" +
                                     output_path.string() + "'");
        }

        const auto rows = read_rows(input_path);
        branchscore::BackendContext backend(selector);
        auto model = branchscore::ModelLoader::load(
            model_path.string(), mmproj_path.string(), backend);
        auto tokenizer = branchscore::GemmaTokenizer::from_gguf(model_path.string());
        const branchscore::Gemma4DecisionEngine engine(model, backend, std::move(tokenizer));

        for (std::size_t index = 0; index < warmup_count; ++index) {
            static_cast<void>(engine.evaluate(rows.front().request));
        }

        std::ofstream output(output_path);
        if (!output) throw std::runtime_error("failed to create output JSONL '" +
                                              output_path.string() + "'");
        output << branchscore::json::stringify(run_json(
            input_path, model_path, mmproj_path, backend, model, warmup_count, rows.size()))
               << '\n';

        std::vector<double> measured_ms;
        std::vector<branchscore::TimingInfo> timings;
        std::vector<branchscore::DecisionResult> results;
        measured_ms.reserve(rows.size());
        timings.reserve(rows.size());
        results.reserve(rows.size());
        const auto wall_started = Clock::now();
        for (const auto & row : rows) {
            const auto started = Clock::now();
            auto result = engine.evaluate(row.request);
            const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
            measured_ms.push_back(elapsed);
            timings.push_back(result.timings);
            results.push_back(std::move(result));
        }
        const auto wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - wall_started).count();
        for (std::size_t index = 0; index < rows.size(); ++index) {
            output << branchscore::json::stringify(
                row_json(rows[index], results[index], measured_ms[index])) << '\n';
        }
        output << branchscore::json::stringify(aggregate_json(measured_ms, timings, wall_ms)) << '\n';
        if (!output) throw std::runtime_error("failed while writing output JSONL");
        std::cout << "Wrote " << rows.size() << " decisions to " << output_path << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "branchscore-bench: " << error.what() << '\n';
        return 1;
    }
}
