#pragma once

#include "branchscore/backend_context.hpp"
#include "branchscore/decision.hpp"
#include "branchscore/decision_limits.hpp"
#include "branchscore/model_loader.hpp"
#include "branchscore/tokenizer.hpp"

namespace branchscore {

class Gemma4DecisionEngine {
public:
    Gemma4DecisionEngine(
        ModelBundle & model,
        BackendContext & backend,
        GemmaTokenizer tokenizer);

    DecisionResult evaluate(const DecisionRequest & request) const;
    // CPU-only admission check; HTTP calls this for every question before any GPU work.
    std::size_t estimate_prefill_positions(const DecisionRequest & request) const;

private:
    ModelBundle & model_;
    BackendContext & backend_;
    GemmaTokenizer tokenizer_;
};

} // namespace branchscore
