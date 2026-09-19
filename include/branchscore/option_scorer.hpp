#pragma once

#include "branchscore/prefill_engine.hpp"

#include <cstddef>
#include <vector>

namespace branchscore {

struct OptionScore {
    std::size_t input_index = 0;
    std::string option_id;
    std::size_t token_count = 0;
    double sum_logprob = 0.0;
    double mean_logprob = 0.0;
    double elapsed_ms = 0.0;
    std::vector<float> token_logprobs;
};

class OptionScorer {
public:
    OptionScorer(ModelBundle & model, BackendContext & backend);

    OptionScore score(PrefillState & state, const OptionTokens & option) const;

private:
    ModelBundle & model_;
    BackendContext & backend_;
};

} // namespace branchscore
