#include "branchscore/option_scorer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace branchscore {

OptionScorer::OptionScorer(ModelBundle & model, BackendContext & backend)
    : model_(model), backend_(backend) {}

OptionScore OptionScorer::score(
    PrefillState & state, const OptionTokens & option) const {
    if (option.ids.empty()) {
        throw std::runtime_error("cannot score an empty option continuation");
    }
    if (!option.boundary_valid) {
        throw std::runtime_error("option token boundary was not validated");
    }
    if (state.prefix_length() + option.ids.size() > state.cache().capacity()) {
        throw std::runtime_error("option continuation exceeds reserved context");
    }

    const auto started = std::chrono::steady_clock::now();
    PrefillEngine engine(model_, backend_);
    state.reset_branch();
    backend_.synchronize();

    OptionScore result;
    result.input_index = option.input_index;
    result.option_id = option.option_id;
    result.token_count = option.ids.size();
    result.token_ids = option.ids;
    result.token_logprobs.reserve(option.ids.size());
    for (std::size_t index = 0; index < option.ids.size(); ++index) {
        result.token_logprobs.push_back(
            engine.score_current(state, option.ids[index]));
        if (index + 1 < option.ids.size()) {
            engine.continue_one(state, option.ids[index]);
        }
    }
    for (const auto value : result.token_logprobs) result.sum_logprob += value;
    result.mean_logprob = result.sum_logprob / result.token_count;
    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return result;
}

OptionSummary summarize_options(std::vector<OptionScore> scores) {
    if (scores.empty()) {
        throw std::runtime_error("cannot summarize an empty option set");
    }
    const auto maximum = std::max_element(
        scores.begin(), scores.end(), [](const OptionScore & left, const OptionScore & right) {
            return left.sum_logprob < right.sum_logprob;
        });
    if (!std::isfinite(maximum->sum_logprob)) {
        throw std::runtime_error("option scores must be finite");
    }
    const auto selected_position = static_cast<std::size_t>(
        maximum - scores.begin());

    double denominator = 0.0;
    for (const auto & score : scores) {
        if (!std::isfinite(score.sum_logprob)) {
            throw std::runtime_error("option scores must be finite");
        }
        denominator += std::exp(score.sum_logprob - maximum->sum_logprob);
    }
    if (!std::isfinite(denominator) || denominator <= 0.0) {
        throw std::runtime_error("option softmax normalization failed");
    }

    OptionSummary result;
    result.scores = std::move(scores);
    result.selected_position = selected_position;
    for (auto & score : result.scores) {
        score.relative_probability =
            std::exp(score.sum_logprob - maximum->sum_logprob) / denominator;
    }
    for (std::size_t index = 0; index < result.scores.size(); ++index) {
        if (index != result.selected_position &&
            result.scores[index].sum_logprob ==
                result.scores[result.selected_position].sum_logprob) {
            result.tie = true;
            break;
        }
    }
    return result;
}

} // namespace branchscore
