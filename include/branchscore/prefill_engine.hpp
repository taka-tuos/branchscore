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
    BackendTiming backend_timing() const noexcept;
    std::size_t graph_node_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    explicit PrefillState(std::unique_ptr<Impl> impl);
    friend class PrefillEngine;
};

class PrefillEngine {
public:
    PrefillEngine(ModelBundle & model, BackendContext & backend);

    PrefillState prefill(
        const std::vector<TokenId> & tokens_before_image,
        const VisualTokens * visual_tokens,
        const std::vector<TokenId> & tokens_after_image) const;

private:
    void continue_one(PrefillState & state, TokenId token) const;
    float score_current(const PrefillState & state, TokenId token) const;

    ModelBundle & model_;
    BackendContext & backend_;
};

} // namespace branchscore
