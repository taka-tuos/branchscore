#include "branchscore/gemma4_decision_engine.hpp"

#include "branchscore/categorical_readout.hpp"
#include "branchscore/gemma4_prompt_renderer.hpp"
#include "branchscore/image_preprocessor.hpp"
#include "branchscore/prefill_engine.hpp"
#include "branchscore/vision_encoder.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace branchscore {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point started) {
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

void validate_request(const DecisionRequest & request) {
    if (request.state.empty()) throw std::runtime_error("state must not be empty");
    if (request.question.empty()) throw std::runtime_error("question must not be empty");
    if (request.options.size() < 2 || request.options.size() > 16) {
        throw std::runtime_error("decision request must contain 2-16 options");
    }
    std::unordered_set<std::string> ids;
    for (const auto & option : request.options) {
        if (option.id.empty()) throw std::runtime_error("option IDs must not be empty");
        if (option.description.empty()) {
            throw std::runtime_error("option descriptions must not be empty");
        }
        if (!ids.insert(option.id).second) {
            throw std::runtime_error("option IDs must be unique");
        }
    }
    if (request.image_path && request.image_path->empty()) {
        throw std::runtime_error("image path must not be empty");
    }
    if (request.image_path && request.image_bytes) {
        throw std::runtime_error("request must use either an image path or image bytes");
    }
    if (request.chat_template_file && request.chat_template_file->empty()) {
        throw std::runtime_error("chat template file name must not be empty");
    }
    if (request.capture_vision_debug && !request.has_image()) {
        throw std::runtime_error("vision debug capture requires an image");
    }
}

} // namespace

Gemma4DecisionEngine::Gemma4DecisionEngine(
    ModelBundle & model,
    BackendContext & backend,
    GemmaTokenizer tokenizer)
    : model_(model), backend_(backend), tokenizer_(std::move(tokenizer)) {}

DecisionResult Gemma4DecisionEngine::evaluate(const DecisionRequest & request) const {
    const auto request_started = Clock::now();
    validate_request(request);

    const auto render_started = Clock::now();
    Gemma4PromptRenderer renderer;
    const auto rendered = renderer.render(
        request.state,
        request.question,
        request.options,
        request.has_image(),
        request.prompt_policy,
        request.chat_template_file,
        tokenizer_.chat_template().has_value());
    const auto prompt_rendering_ms = elapsed_ms(render_started);

    const auto tokenization_started = Clock::now();
    const auto all_ids = tokenizer_.tokenize(rendered.text, false, true);
    const auto image_id = tokenizer_.find_token("<|image|>");
    const auto image_count = image_id == std::nullopt
        ? 0U
        : static_cast<std::size_t>(std::count(all_ids.begin(), all_ids.end(), *image_id));
    if (request.has_image() && image_count != 1) {
        throw std::runtime_error(
            "Gemma 4 image request must render exactly one <|image|> token");
    }
    if (!request.has_image() && image_count != 0) {
        throw std::runtime_error(
            "text-only request unexpectedly contains an <|image|> token");
    }
    const auto image_position = image_count == 0
        ? all_ids.end()
        : std::find(all_ids.begin(), all_ids.end(), *image_id);
    const std::size_t image_index = image_position == all_ids.end()
        ? all_ids.size()
        : static_cast<std::size_t>(image_position - all_ids.begin());
    std::vector<TokenId> tokens_before(all_ids.begin(), all_ids.begin() + image_index);
    std::vector<TokenId> tokens_after;
    if (image_position != all_ids.end()) {
        tokens_after.assign(image_position + 1, all_ids.end());
    }

    std::vector<AnswerToken> answer_tokens;
    answer_tokens.reserve(rendered.answer_slots.size());
    std::unordered_set<TokenId> answer_token_ids;
    for (const auto & slot : rendered.answer_slots) {
        const auto answer = tokenizer_.tokenize_answer_label(rendered.text, slot.label);
        if (!answer.boundary_valid || !answer_token_ids.insert(answer.id).second) {
            throw std::runtime_error(
                "categorical answer labels must map to distinct valid tokens");
        }
        answer_tokens.push_back(answer);
    }
    const auto tokenization_ms = elapsed_ms(tokenization_started);

    std::unique_ptr<PreparedImage> prepared_image;
    std::unique_ptr<VisualTokens> visual_tokens;
    std::optional<VisionDebugInfo> vision_debug;
    double image_preprocessing_ms = 0.0;
    double vision_ms = 0.0;
    if (request.has_image()) {
        const auto image_started = Clock::now();
        if (request.image_path) {
            prepared_image = std::make_unique<PreparedImage>(ImagePreprocessor::load(
                *request.image_path, model_.vision_config()));
        } else {
            prepared_image = std::make_unique<PreparedImage>(ImagePreprocessor::load_encoded(
                request.image_bytes->data(),
                request.image_bytes->size(),
                model_.vision_config()));
        }
        image_preprocessing_ms = elapsed_ms(image_started);

        const auto vision_started = Clock::now();
        visual_tokens = std::make_unique<VisualTokens>(
            VisionEncoder(model_, backend_).encode(*prepared_image));
        if (request.capture_vision_debug) {
            vision_debug = VisionDebugInfo{
                visual_tokens->token_count(),
                visual_tokens->embedding_length(),
                visual_tokens->download(),
            };
        }
        vision_ms = elapsed_ms(vision_started);
    }

    const auto prefill_started = Clock::now();
    PrefillEngine prefill(model_, backend_);
    auto prefill_state = prefill.prefill(
        tokens_before,
        visual_tokens.get(),
        tokens_after);
    const auto prefill_ms = elapsed_ms(prefill_started);
    const auto prefill_backend_timing = prefill_state.backend_timing();
    const auto prefill_graph_node_count = prefill_state.graph_node_count();

    const auto readout_started = Clock::now();
    std::vector<TokenId> answer_ids;
    answer_ids.reserve(answer_tokens.size());
    for (const auto & answer : answer_tokens) answer_ids.push_back(answer.id);
    const auto logits = gather_categorical_logits(
        prefill_state, answer_ids, backend_);
    const auto readout_ms = elapsed_ms(readout_started);

    const auto normalization_started = Clock::now();
    const auto summary = summarize_categorical(logits.raw_scores);
    const auto normalization_ms = elapsed_ms(normalization_started);

    DecisionResult result;
    result.option_scores.reserve(rendered.answer_slots.size());
    for (std::size_t index = 0; index < rendered.answer_slots.size(); ++index) {
        const auto & slot = rendered.answer_slots[index];
        result.option_scores.push_back(OptionScore{
            slot.input_index,
            slot.option_id,
            slot.label,
            answer_tokens[index].id,
            static_cast<double>(logits.raw_scores[index]),
            summary.relative_probabilities[index],
        });
    }
    result.selected_index = result.option_scores[summary.selected_index].input_index;
    result.selected_id = result.option_scores[summary.selected_index].option_id;
    result.exact_tie = summary.exact_tie;
    result.scoring_basis = "answer_slot_logit";
    result.readout_id = "gemma4-next-token-categorical-v1";
    result.terminator_scored = false;
    result.prompt_format = rendered.format;
    result.rendered_prompt_identity = rendered.identity;
    result.rendered_prompt_token_ids = all_ids;
    result.vision_debug = std::move(vision_debug);
    result.timings.prompt_rendering_ms = prompt_rendering_ms;
    result.timings.tokenization_ms = tokenization_ms;
    result.timings.image_preprocessing_ms = image_preprocessing_ms;
    result.timings.vision_ms = vision_ms;
    if (visual_tokens) {
        const auto timing = visual_tokens->backend_timing();
        result.timings.vision_backend_copy_ms = timing.copy_ms;
        result.timings.vision_synchronization_ms = timing.synchronization_ms;
        result.timings.vision_graph_node_count = visual_tokens->graph_node_count();
    }
    result.timings.prefill_ms = prefill_ms;
    result.timings.prefill_backend_copy_ms = prefill_backend_timing.copy_ms;
    result.timings.prefill_synchronization_ms = prefill_backend_timing.synchronization_ms;
    result.timings.prefill_graph_node_count = prefill_graph_node_count;
    result.timings.readout_ms = readout_ms;
    result.timings.readout_backend_copy_ms = logits.backend_timing.copy_ms;
    result.timings.readout_synchronization_ms = logits.backend_timing.synchronization_ms;
    result.timings.readout_graph_node_count = logits.graph_node_count;
    result.timings.normalization_ms = normalization_ms;
    result.timings.request_total_ms = elapsed_ms(request_started);
    return result;
}

} // namespace branchscore
