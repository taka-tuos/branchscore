#include "branchscore/option_scorer.hpp"

#include <chrono>
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

} // namespace branchscore
