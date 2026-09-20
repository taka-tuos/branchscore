#pragma once

#include "branchscore/prefill_engine.hpp"
#include "branchscore/decision.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace branchscore {

struct OptionSummary {
    std::vector<OptionScore> scores;
    std::size_t selected_position = 0;
    bool tie = false;
};

OptionSummary summarize_options(std::vector<OptionScore> scores);

class OptionScorer {
public:
    OptionScorer(ModelBundle & model, BackendContext & backend);

    OptionScore score(PrefillState & state, const OptionTokens & option) const;

private:
    ModelBundle & model_;
    BackendContext & backend_;
};

} // namespace branchscore
