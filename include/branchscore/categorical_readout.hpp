#pragma once

#include "branchscore/backend_context.hpp"
#include "branchscore/prefill_engine.hpp"
#include "branchscore/tokenizer.hpp"

#include <cstddef>
#include <vector>

namespace branchscore {

struct CategoricalLogits {
    std::vector<float> raw_scores;
    BackendTiming backend_timing;
    std::size_t graph_node_count = 0;
};

struct CategoricalSummary {
    std::vector<double> relative_probabilities;
    std::size_t selected_index = 0;
    bool exact_tie = false;
};

CategoricalLogits gather_categorical_logits(
    const PrefillState & state,
    const std::vector<TokenId> & answer_token_ids,
    BackendContext & backend);

CategoricalSummary summarize_categorical(const std::vector<float> & raw_scores);

} // namespace branchscore
