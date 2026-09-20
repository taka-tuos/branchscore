#include "branchscore/state_cache.hpp"

#include "ggml-alloc.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace branchscore {

struct StateCache::Impl {
    ~Impl() {
        if (buffer != nullptr) ggml_backend_buffer_free(buffer);
        if (ctx != nullptr) ggml_free(ctx);
    }

    TextModelConfig config;
    std::size_t capacity = 0;
    std::size_t prefix_length = 0;
    std::size_t cursor = 0;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_tensor *> keys;
    std::vector<ggml_tensor *> values;
};

StateCache::StateCache(
    const TextModelConfig & config,
    std::size_t capacity,
    BackendContext & backend)
    : impl_(std::make_unique<Impl>()) {
    if (capacity == 0 || capacity > config.context_length) {
        throw std::runtime_error("state-cache capacity is outside the model context");
    }
    if (config.shared_kv_layers >= config.block_count) {
        throw std::runtime_error("invalid Gemma 4 shared-KV layer count");
    }
    impl_->config = config;
    impl_->capacity = capacity;
    const auto stored_layers = config.block_count - config.shared_kv_layers;
    ggml_init_params params{
        (2U * stored_layers + 1U) * ggml_tensor_overhead(), nullptr, true,
    };
    impl_->ctx = ggml_init(params);
    if (impl_->ctx == nullptr) {
        throw std::runtime_error("failed to create state-cache tensor context");
    }
    impl_->keys.reserve(stored_layers);
    impl_->values.reserve(stored_layers);
    for (std::uint32_t layer = 0; layer < stored_layers; ++layer) {
        auto * key = ggml_new_tensor_2d(
            impl_->ctx, GGML_TYPE_F32, config.key_width(layer), capacity);
        auto * value = ggml_new_tensor_2d(
            impl_->ctx, GGML_TYPE_F32, config.value_width(layer), capacity);
        ggml_format_name(key, "cache_k_%u", layer);
        ggml_format_name(value, "cache_v_%u", layer);
        impl_->keys.push_back(key);
        impl_->values.push_back(value);
    }
    impl_->buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        impl_->ctx, backend.buffer_type());
    if (impl_->buffer == nullptr) {
        throw std::runtime_error("failed to allocate state cache on selected backend");
    }
}

StateCache::~StateCache() = default;
StateCache::StateCache(StateCache &&) noexcept = default;
StateCache & StateCache::operator=(StateCache &&) noexcept = default;

std::size_t StateCache::capacity() const noexcept { return impl_->capacity; }
std::size_t StateCache::prefix_length() const noexcept { return impl_->prefix_length; }
std::size_t StateCache::cursor() const noexcept { return impl_->cursor; }

std::uint32_t StateCache::source_layer(std::uint32_t layer) const {
    if (layer >= impl_->config.block_count) {
        throw std::out_of_range("Gemma 4 layer index is out of range");
    }
    const auto stored_layers =
        impl_->config.block_count - impl_->config.shared_kv_layers;
    if (layer < stored_layers) return layer;
    const auto source = stored_layers -
        (impl_->config.is_sliding_window(layer) ? 2U : 1U);
    if (impl_->config.is_sliding_window(source) !=
        impl_->config.is_sliding_window(layer)) {
        throw std::runtime_error("shared-KV source has a different attention type");
    }
    return source;
}

ggml_tensor * StateCache::key(std::uint32_t layer) const {
    return impl_->keys.at(source_layer(layer));
}

ggml_tensor * StateCache::value(std::uint32_t layer) const {
    return impl_->values.at(source_layer(layer));
}

void StateCache::freeze_prefix(std::size_t length) {
    if (length == 0 || length > capacity()) {
        throw std::runtime_error("prefix length is outside state-cache capacity");
    }
    if (impl_->prefix_length != 0) {
        throw std::runtime_error("state-cache prefix is already frozen");
    }
    impl_->prefix_length = length;
    impl_->cursor = length;
}

} // namespace branchscore
