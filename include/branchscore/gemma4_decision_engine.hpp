#pragma once

#include "branchscore/backend_context.hpp"
#include "branchscore/decision.hpp"
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

private:
    ModelBundle & model_;
    BackendContext & backend_;
    GemmaTokenizer tokenizer_;
};

} // namespace branchscore
