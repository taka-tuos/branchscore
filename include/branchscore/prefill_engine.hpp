#pragma once

#include "branchscore/backend_context.hpp"
#include "branchscore/model_loader.hpp"
#include "branchscore/state_cache.hpp"
#include "branchscore/tokenizer.hpp"
#include "branchscore/vision_encoder.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace branchscore {

class OptionScorer;

class PrefillState {
public:
    ~PrefillState();
    PrefillState(PrefillState &&) noexcept;
    PrefillState & operator=(PrefillState &&) noexcept;

    PrefillState(const PrefillState &) = delete;
    PrefillState & operator=(const PrefillState &) = delete;

    std::size_t prefix_length() const noexcept;
    StateCache & cache() noexcept;
    const StateCache & cache() const noexcept;
    ggml_tensor * logits() const noexcept;
    std::vector<float> download_logits() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    explicit PrefillState(std::unique_ptr<Impl> impl);
    void reset_branch();
    friend class PrefillEngine;
    friend class OptionScorer;
};

class PrefillEngine {
public:
    PrefillEngine(ModelBundle & model, BackendContext & backend);

    PrefillState prefill(
        const std::vector<TokenId> & tokens_before_image,
        const VisualTokens * visual_tokens,
        const std::vector<TokenId> & tokens_after_image,
        std::size_t maximum_option_tokens) const;

private:
    void continue_one(PrefillState & state, TokenId token) const;
    float score_current(const PrefillState & state, TokenId token) const;

    ModelBundle & model_;
    BackendContext & backend_;

    friend class OptionScorer;
};

} // namespace branchscore
