#include "branchscore/backend_context.hpp"
#include "branchscore/decision.hpp"
#include "branchscore/gemma4_decision_engine.hpp"
#include "branchscore/model_loader.hpp"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

double gib(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

void print_devices() {
    const auto devices = branchscore::BackendContext::available_devices();
    if (devices.empty()) {
        std::cout << "No ggml backend devices found\n";
        return;
    }

    for (const auto & device : devices) {
        std::cout << device.name << "\tfamily=" << device.family
                  << "\ttype=" << device.type;
        if (device.memory_total != 0) {
            std::cout << "\tmemory=" << std::fixed << std::setprecision(2)
                      << gib(device.memory_free) << "/" << gib(device.memory_total)
                      << " GiB free";
        }
        std::cout << "\t" << device.description << '\n';
    }
}

branchscore::DecisionOption parse_option(const std::string & value) {
    const auto separator = value.find('=');
    if (separator == std::string::npos || separator == 0 || separator + 1 >= value.size()) {
        throw std::runtime_error("--option expects ID=DESCRIPTION");
    }
    return {value.substr(0, separator), value.substr(separator + 1)};
}

std::string format_ids(const std::vector<branchscore::TokenId> & ids) {
    std::ostringstream output;
    for (std::size_t index = 0; index < ids.size(); ++index) {
        if (index != 0) output << ',';
        output << ids[index];
    }
    return output.str();
}

void write_vision_dump(
    const std::string & path,
    const branchscore::VisionDebugInfo & debug) {
    std::ofstream dump(path, std::ios::binary);
    if (!dump) throw std::runtime_error("failed to open vision dump '" + path + "'");
    const std::int32_t header[] = {
        static_cast<std::int32_t>(debug.token_count),
        static_cast<std::int32_t>(debug.embedding_length),
    };
    dump.write(reinterpret_cast<const char *>(header), sizeof(header));
    dump.write(
        reinterpret_cast<const char *>(debug.values.data()),
        static_cast<std::streamsize>(debug.values.size() * sizeof(float)));
    if (!dump) throw std::runtime_error("failed to write vision dump");
}

} // namespace

int main(int argc, char ** argv) {
    try {
        std::string selector = "auto";
        std::string model_path;
        std::string mmproj_path;
        std::string image_path;
        std::string vision_dump_path;
        std::string chat_template_file;
        std::string state;
        std::string question;
        std::vector<branchscore::DecisionOption> options;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--list-backends") {
                print_devices();
                return 0;
            }
            if (arg == "--backend" && i + 1 < argc) {
                selector = argv[++i];
                continue;
            }
            if (arg == "--model" && i + 1 < argc) {
                model_path = argv[++i];
                continue;
            }
            if (arg == "--mmproj" && i + 1 < argc) {
                mmproj_path = argv[++i];
                continue;
            }
            if (arg == "--image" && i + 1 < argc) {
                image_path = argv[++i];
                continue;
            }
            if (arg == "--vision-dump" && i + 1 < argc) {
                vision_dump_path = argv[++i];
                continue;
            }
            if (arg == "--chat-template-file" && i + 1 < argc) {
                chat_template_file = argv[++i];
                continue;
            }
            if (arg == "--state" && i + 1 < argc) {
                state = argv[++i];
                continue;
            }
            if (arg == "--question" && i + 1 < argc) {
                question = argv[++i];
                continue;
            }
            if (arg == "--option" && i + 1 < argc) {
                options.push_back(parse_option(argv[++i]));
                continue;
            }
            if (arg == "--help") {
                std::cout
                    << "Usage: branchscore [--list-backends] [--backend NAME]\n"
                    << "                   --model FILE --mmproj FILE\n"
                    << "                   --state TEXT --question TEXT\n"
                    << "                   --option ID=DESCRIPTION [--option ID=DESCRIPTION ...]\n"
                    << "                   [--image FILE] [--vision-dump FILE]\n"
                    << "                   [--chat-template-file FILE]\n";
                return 0;
            }
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }

        if (model_path.empty() != mmproj_path.empty()) {
            throw std::runtime_error("--model and --mmproj must be supplied together");
        }
        if (model_path.empty()) {
            if (!image_path.empty() || !vision_dump_path.empty() ||
                !chat_template_file.empty() ||
                !state.empty() || !question.empty() || !options.empty()) {
                throw std::runtime_error("decision arguments require --model and --mmproj");
            }
            branchscore::BackendContext backend(selector);
            backend.synchronize();
            return 0;
        }

        branchscore::BackendContext backend(selector);
        auto model = branchscore::ModelLoader::load(model_path, mmproj_path, backend);
        const auto & text = model.text_config();
        const auto & vision = model.vision_config();
        std::cout << "Loaded " << text.name << ": " << text.block_count
                  << " text blocks, width " << text.embedding_length
                  << ", vocab " << text.vocabulary_size << ", "
                  << model.text_tensor_count() << " tensors ("
                  << std::fixed << std::setprecision(2)
                  << gib(model.text_weight_bytes()) << " GiB)\n";
        std::cout << "Loaded " << vision.projector_type << ": "
                  << vision.block_count << " vision blocks, width "
                  << vision.embedding_length << " -> "
                  << vision.projection_length << ", "
                  << model.vision_tensor_count() << " tensors ("
                  << gib(model.vision_weight_bytes()) << " GiB)\n";

        auto tokenizer = branchscore::GemmaTokenizer::from_gguf(model_path);
        branchscore::DecisionRequest request;
        request.state = std::move(state);
        request.question = std::move(question);
        request.options = std::move(options);
        if (!image_path.empty()) request.image_path = std::move(image_path);
        if (!vision_dump_path.empty()) request.capture_vision_debug = true;
        if (!chat_template_file.empty()) {
            request.chat_template_file = std::move(chat_template_file);
        }

        const branchscore::Gemma4DecisionEngine engine(
            model, backend, std::move(tokenizer));
        const auto result = engine.evaluate(request);

        if (!vision_dump_path.empty()) {
            if (!result.vision_debug) {
                throw std::runtime_error("vision debug output was not produced");
            }
            write_vision_dump(vision_dump_path, *result.vision_debug);
            std::cout << "Wrote vision dump: " << vision_dump_path << '\n';
        }

        std::cout << std::defaultfloat << std::setprecision(9);
        std::cout << "Prompt renderer: " << result.prompt_format.renderer_id
                  << " model_family=" << result.prompt_format.model_family
                  << " effective_source=" << result.prompt_format.effective_source
                  << " reasoning_policy="
                  << branchscore::reasoning_policy_name(
                         result.prompt_format.prompt_policy.reasoning)
                  << '\n';
        if (result.prompt_format.requested_template_file) {
            std::cout << "Requested chat template: "
                      << *result.prompt_format.requested_template_file
                      << " applied="
                      << (result.prompt_format.requested_template_applied ? "true" : "false")
                      << '\n';
        } else {
            std::cout << "Requested chat template: none applied=false\n";
        }
        std::cout << "GGUF tokenizer.chat_template: present="
                  << (result.prompt_format.gguf_chat_template_present ? "true" : "false")
                  << " used="
                  << (result.prompt_format.gguf_chat_template_used ? "true" : "false")
                  << '\n';
        std::cout << "Rendered prompt identity: "
                  << result.rendered_prompt_identity << '\n'
                  << "Rendered prefix tokens: "
                  << result.rendered_prefix_token_ids.size() << '\n'
                  << "Prefix IDs: "
                  << format_ids(result.rendered_prefix_token_ids) << '\n';
        for (const auto & score : result.option_scores) {
            std::cout << "Option " << score.option_id
                      << " IDs: " << format_ids(score.token_ids) << '\n';
            std::cout << "Score " << score.option_id
                      << " index=" << score.input_index
                      << " tokens=" << score.token_count
                      << " sum_logprob=" << score.sum_logprob
                      << " mean_logprob=" << score.mean_logprob
                      << " relative_probability=" << score.relative_probability
                      << " elapsed_ms=" << score.elapsed_ms << '\n';
        }
        std::cout << "scoring_basis=" << result.scoring_basis
                  << " terminator_scored="
                  << (result.terminator_scored ? "true" : "false") << '\n'
                  << "Selected: " << result.selected_id
                  << " index=" << result.selected_index
                  << " exact_tie=" << (result.exact_tie ? "true" : "false") << '\n';
        std::cout << "Timing ms: prompt_rendering="
                  << result.timings.prompt_rendering_ms
                  << " tokenization=" << result.timings.tokenization_ms
                  << " image_preprocessing=" << result.timings.image_preprocessing_ms
                  << " vision=" << result.timings.vision_ms
                  << " prefill=" << result.timings.prefill_ms
                  << " score_total=" << result.timings.score_total_ms
                  << " normalization=" << result.timings.normalization_ms
                  << " request_total=" << result.timings.request_total_ms << '\n';
        std::cout << "Relative probabilities are conditional on the supplied option set;"
                     " they are not calibrated confidence.\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "branchscore: " << error.what() << '\n';
        return 1;
    }
}
