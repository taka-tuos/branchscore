#include "branchscore/gemma4_decision_engine.hpp"

#include "branchscore/gemma4_prompt_renderer.hpp"
#include "branchscore/image_preprocessor.hpp"
#include "branchscore/option_scorer.hpp"
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
    if (request.chat_template_file && request.chat_template_file->empty()) {
        throw std::runtime_error("chat template file name must not be empty");
    }
    if (request.capture_vision_debug && !request.image_path) {
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
        request.image_path.has_value(),
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
    if (request.image_path.has_value() && image_count != 1) {
        throw std::runtime_error(
            "Gemma 4 image request must render exactly one <|image|> token");
    }
    if (!request.image_path.has_value() && image_count != 0) {
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

    std::vector<OptionTokens> option_tokens;
    option_tokens.reserve(request.options.size());
    std::size_t maximum_option_tokens = 0;
    for (std::size_t index = 0; index < request.options.size(); ++index) {
        option_tokens.push_back(tokenizer_.tokenize_option(
            rendered.text,
            request.options[index].id,
            index,
            request.options[index].description));
        maximum_option_tokens = std::max(
            maximum_option_tokens, option_tokens.back().ids.size());
    }
    const auto tokenization_ms = elapsed_ms(tokenization_started);

    std::unique_ptr<PreparedImage> prepared_image;
    std::unique_ptr<VisualTokens> visual_tokens;
    std::optional<VisionDebugInfo> vision_debug;
    double image_preprocessing_ms = 0.0;
    double vision_ms = 0.0;
    if (request.image_path.has_value()) {
        const auto image_started = Clock::now();
        prepared_image = std::make_unique<PreparedImage>(ImagePreprocessor::load(
            *request.image_path, model_.vision_config()));
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
        tokens_after,
        maximum_option_tokens);
    const auto prefill_ms = elapsed_ms(prefill_started);

    OptionScorer scorer(model_, backend_);
    const auto score_started = Clock::now();
    std::vector<OptionScore> scores;
    scores.reserve(option_tokens.size());
    for (const auto & option : option_tokens) {
        scores.push_back(scorer.score(prefill_state, option));
    }
    const auto score_total_ms = elapsed_ms(score_started);

    const auto normalization_started = Clock::now();
    const auto summary = summarize_options(std::move(scores));
    const auto normalization_ms = elapsed_ms(normalization_started);

    DecisionResult result;
    result.option_scores = summary.scores;
    result.selected_index = result.option_scores[summary.selected_position].input_index;
    result.selected_id = result.option_scores[summary.selected_position].option_id;
    result.exact_tie = summary.tie;
    result.scoring_basis = "sum_logprob";
    result.terminator_scored = false;
    result.prompt_format = rendered.format;
    result.rendered_prompt_identity = rendered.identity;
    result.rendered_prefix_token_ids = all_ids;
    result.vision_debug = std::move(vision_debug);
    result.timings.prompt_rendering_ms = prompt_rendering_ms;
    result.timings.tokenization_ms = tokenization_ms;
    result.timings.image_preprocessing_ms = image_preprocessing_ms;
    result.timings.vision_ms = vision_ms;
    result.timings.prefill_ms = prefill_ms;
    result.timings.option_scoring_ms.reserve(result.option_scores.size());
    for (const auto & score : result.option_scores) {
        result.timings.option_scoring_ms.push_back(score.elapsed_ms);
    }
    result.timings.score_total_ms = score_total_ms;
    result.timings.normalization_ms = normalization_ms;
    result.timings.request_total_ms = elapsed_ms(request_started);
    return result;
}

} // namespace branchscore
