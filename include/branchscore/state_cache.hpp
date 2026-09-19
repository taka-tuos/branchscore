#pragma once

#include "branchscore/backend_context.hpp"
#include "branchscore/model_loader.hpp"

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace branchscore {

class StateCache {
public:
    StateCache(
        const TextModelConfig & config,
        std::size_t capacity,
        BackendContext & backend);
    ~StateCache();

    StateCache(const StateCache &) = delete;
    StateCache & operator=(const StateCache &) = delete;
    StateCache(StateCache &&) noexcept;
    StateCache & operator=(StateCache &&) noexcept;

    std::size_t capacity() const noexcept;
    std::size_t prefix_length() const noexcept;
    std::size_t cursor() const noexcept;

    std::uint32_t source_layer(std::uint32_t layer) const;
    ggml_tensor * key(std::uint32_t layer) const;
    ggml_tensor * value(std::uint32_t layer) const;

    void freeze_prefix(std::size_t length);
    void reset_branch() noexcept;
    void advance(std::size_t token_count);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace branchscore
