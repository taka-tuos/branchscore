#include "branchscore/backend_context.hpp"
#include "branchscore/chat_template.hpp"
#include "branchscore/image_preprocessor.hpp"
#include "branchscore/model_loader.hpp"
#include "branchscore/option_scorer.hpp"
#include "branchscore/vision_encoder.hpp"
#include "branchscore/prefill_engine.hpp"
#include "branchscore/tokenizer.hpp"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
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

struct CliOption {
    std::string id;
    std::string description;
};

CliOption parse_option(const std::string & value) {
    const auto separator = value.find('=');
    if (separator == std::string::npos || separator == 0 || separator + 1 >= value.size()) {
        throw std::runtime_error("--option expects ID=DESCRIPTION");
    }
    return {value.substr(0, separator), value.substr(separator + 1)};
}

std::string read_state_user_prompt(
    const std::string & state, const std::string & question) {
    return "State:\n" + state + "\n\nQuestion:\n" + question;
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
    const std::string & path, const branchscore::VisualTokens & embeddings) {
    std::ofstream dump(path, std::ios::binary);
    if (!dump) {
        throw std::runtime_error("failed to open vision dump '" + path + "'");
    }
    const std::int32_t header[] = {
        static_cast<std::int32_t>(embeddings.token_count()),
        static_cast<std::int32_t>(embeddings.embedding_length()),
    };
    dump.write(reinterpret_cast<const char *>(header), sizeof(header));
    const auto values = embeddings.download();
    dump.write(
        reinterpret_cast<const char *>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
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
        std::vector<CliOption> options;
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
                    << "                   [--image FILE] [--chat-template-file FILE]\n"
                    << "                   [--vision-dump FILE]\n";
                return 0;
            }
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }

        branchscore::BackendContext backend(selector);
        std::cout << "Initialized " << backend.device().name
                  << " (" << backend.device().family << ", "
                  << backend.device().type << ")\n";

        if (model_path.empty() != mmproj_path.empty()) {
            throw std::runtime_error("--model and --mmproj must be supplied together");
        }
        if (!model_path.empty() && (state.empty() || question.empty())) {
            throw std::runtime_error("--state and --question are required for decision scoring");
        }
        if (!model_path.empty() && (options.size() < 2 || options.size() > 16)) {
            throw std::runtime_error("provide 2-16 repeated --option ID=DESCRIPTION values");
        }
        if (!model_path.empty()) {
            std::unordered_set<std::string> option_ids;
            for (const auto & option : options) {
                if (!option_ids.insert(option.id).second) {
                    throw std::runtime_error("option IDs must be unique");
                }
            }
        }
        if (!chat_template_file.empty() && model_path.empty()) {
            throw std::runtime_error("--chat-template-file requires --model");
        }
        if (!image_path.empty() && model_path.empty()) {
            throw std::runtime_error("--image requires --model and --mmproj");
        }
        if (!vision_dump_path.empty() && image_path.empty()) {
            throw std::runtime_error("--vision-dump requires --image");
        }
        if (!model_path.empty()) {
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

            const auto tokenizer = branchscore::GemmaTokenizer::from_gguf(model_path);
            const auto chat_template = chat_template_file.empty()
                ? branchscore::ChatTemplate::from_source(
                    tokenizer.chat_template(), "tokenizer.chat_template")
                : branchscore::ChatTemplate::from_file(chat_template_file);
            const branchscore::ChatPrompt prompt{
                "Use the supplied state to answer the question. Return only the answer.",
                read_state_user_prompt(state, question), !image_path.empty(),
            };
            const auto rendered = chat_template.render(prompt, true);
            const auto all_ids = tokenizer.tokenize(rendered, false, true);
            const auto image_id = tokenizer.find_token("<|image|>");
            const auto image_position = image_id == std::nullopt
                ? all_ids.end()
                : std::find(all_ids.begin(), all_ids.end(), *image_id);
            if (!image_path.empty() && image_position == all_ids.end()) {
                throw std::runtime_error("rendered chat template has no <|image|> marker");
            }
            if (image_path.empty() && image_position != all_ids.end()) {
                throw std::runtime_error("rendered chat template unexpectedly has an image marker");
            }
            const std::size_t image_index =
                image_position == all_ids.end()
                    ? all_ids.size()
                    : static_cast<std::size_t>(image_position - all_ids.begin());
            std::vector<branchscore::TokenId> tokens_before(
                all_ids.begin(), all_ids.begin() + image_index);
            std::vector<branchscore::TokenId> tokens_after;
            if (image_position != all_ids.end()) {
                tokens_after.assign(image_position + 1, all_ids.end());
            }
            std::cout << "Chat template: " << chat_template.origin() << '\n'
                      << "Rendered prefix tokens: " << all_ids.size() << '\n'
                      << "Prefix IDs: " << format_ids(all_ids) << '\n';

            std::vector<branchscore::OptionTokens> option_tokens;
            option_tokens.reserve(options.size());
            std::size_t maximum_option_tokens = 0;
            for (std::size_t index = 0; index < options.size(); ++index) {
                option_tokens.push_back(tokenizer.tokenize_option(
                    rendered, options[index].id, index, options[index].description));
                maximum_option_tokens = std::max(
                    maximum_option_tokens, option_tokens.back().ids.size());
                std::cout << "Option " << options[index].id << " IDs: "
                          << format_ids(option_tokens.back().ids) << '\n';
            }

            std::unique_ptr<branchscore::PreparedImage> prepared_image;
            std::unique_ptr<branchscore::VisualTokens> visual_tokens;
            double vision_ms = 0.0;
            if (!image_path.empty()) {
                const auto vision_started = std::chrono::steady_clock::now();
                prepared_image = std::make_unique<branchscore::PreparedImage>(
                    branchscore::ImagePreprocessor::load(image_path, vision));
                visual_tokens = std::make_unique<branchscore::VisualTokens>(
                    branchscore::VisionEncoder(model, backend).encode(*prepared_image));
                vision_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - vision_started).count();
                std::cout << "Prepared image: " << prepared_image->width << "x"
                          << prepared_image->height << ", "
                          << visual_tokens->token_count() << " visual tokens\n";
                if (!vision_dump_path.empty()) {
                    write_vision_dump(vision_dump_path, *visual_tokens);
                    std::cout << "Wrote vision dump: " << vision_dump_path << '\n';
                }
            }

            const auto prefill_started = std::chrono::steady_clock::now();
            branchscore::PrefillEngine prefill(model, backend);
            auto prefill_state = prefill.prefill(
                tokens_before, visual_tokens.get(), tokens_after, maximum_option_tokens);
            backend.synchronize();
            const auto prefill_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - prefill_started).count();
            branchscore::OptionScorer scorer(model, backend);
            const auto score_started = std::chrono::steady_clock::now();
            std::vector<branchscore::OptionScore> scores;
            scores.reserve(option_tokens.size());
            for (const auto & option : option_tokens) {
                scores.push_back(scorer.score(prefill_state, option));
            }
            const auto score_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - score_started).count();
            const auto summary = branchscore::summarize_options(std::move(scores));
            std::cout << std::defaultfloat << std::setprecision(9);
            for (const auto & score : summary.scores) {
                std::cout << "Score " << score.option_id
                          << " sum_logprob=" << score.sum_logprob
                          << " mean_logprob=" << score.mean_logprob
                          << " probability=" << score.relative_probability
                          << " elapsed_ms=" << score.elapsed_ms << '\n';
            }
            std::cout << "Selected: "
                      << summary.scores[summary.selected_position].option_id
                      << " tie=" << (summary.tie ? "true" : "false") << '\n'
                      << "Timing ms: vision=" << vision_ms
                      << " prefill=" << prefill_ms
                      << " score=" << score_ms
                      << " total=" << (vision_ms + prefill_ms + score_ms) << '\n';
            return 0;
        }
        backend.synchronize();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "branchscore: " << error.what() << '\n';
        return 1;
    }
}
