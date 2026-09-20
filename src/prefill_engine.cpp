#include "branchscore/prefill_engine.hpp"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace branchscore {
namespace {

constexpr std::size_t max_graph_nodes = 8192;

ggml_tensor * require_text_tensor(ModelBundle & model, const std::string & name) {
    auto * tensor = model.text_tensor(name);
    if (tensor == nullptr) throw std::runtime_error("missing text tensor: " + name);
    return tensor;
}

ggml_tensor * rms_norm(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * weight,
    float epsilon) {
    auto * result = ggml_rms_norm(ctx, input, epsilon);
    return weight == nullptr ? result : ggml_mul(ctx, result, weight);
}

std::string layer_name(std::uint32_t layer, const char * suffix) {
    return "blk." + std::to_string(layer) + "." + suffix;
}

ggml_tensor * layer_tensor(
    ModelBundle & model,
    std::uint32_t layer,
    const char * suffix) {
    return require_text_tensor(model, layer_name(layer, suffix));
}

ggml_tensor * build_attention(
    ggml_context * ctx,
    ggml_tensor * query,
    ggml_tensor * key,
    ggml_tensor * value,
    ggml_tensor * mask) {
    query = ggml_permute(ctx, query, 0, 2, 1, 3);
    key = ggml_permute(ctx, key, 0, 2, 1, 3);
    value = ggml_permute(ctx, value, 0, 2, 1, 3);
    value = ggml_cont(ctx, ggml_transpose(ctx, value));
    auto * scores = ggml_mul_mat(ctx, key, query);
    scores = ggml_soft_max_ext(ctx, scores, mask, 1.0F, 0.0F);
    auto * attended = ggml_mul_mat(ctx, value, scores);
    attended = ggml_permute(ctx, attended, 0, 2, 1, 3);
    return ggml_cont_2d(
        ctx, attended, attended->ne[0] * attended->ne[1],
        attended->ne[2] * attended->ne[3]);
}

ggml_tensor * append_tokens(
    ggml_context * ctx,
    ggml_tensor * current,
    ggml_tensor * additional) {
    return current == nullptr ? additional : ggml_concat(ctx, current, additional, 1);
}

std::vector<float> causal_mask(
    std::size_t query_start,
    std::size_t query_count,
    std::size_t key_count,
    std::size_t window) {
    std::vector<float> result(query_count * key_count);
    const float blocked = -std::numeric_limits<float>::infinity();
    for (std::size_t query = 0; query < query_count; ++query) {
        const std::size_t absolute_query = query_start + query;
        for (std::size_t key = 0; key < key_count; ++key) {
            const bool after_query = key > absolute_query;
            const bool before_window =
                window != 0 && key + window <= absolute_query;
            result.at(query * key_count + key) =
                after_query || before_window ? blocked : 0.0F;
        }
    }
    return result;
}

ggml_tensor * cache_current_kv(
    ggml_context * ctx,
    ggml_tensor * current,
    ggml_tensor * cache,
    std::size_t width,
    std::size_t token_count,
    std::size_t cache_start) {
    auto * destination = ggml_view_2d(
        ctx, cache, width, token_count, cache->nb[1], cache_start * cache->nb[1]);
    current = ggml_reshape_2d(ctx, current, width, token_count);
    return ggml_cpy(ctx, current, destination);
}

ggml_tensor * build_layer(
    ggml_context * ctx,
    ggml_cgraph * graph,
    ModelBundle & model,
    StateCache & cache,
    const TextModelConfig & config,
    std::uint32_t layer,
    std::size_t query_count,
    std::size_t cache_start,
    std::size_t cache_count,
    ggml_tensor * positions,
    ggml_tensor * full_mask,
    ggml_tensor * sliding_mask,
    ggml_tensor * per_layer_inputs,
    ggml_tensor * input) {
    const bool sliding = config.is_sliding_window(layer);
    const std::uint32_t head_size = sliding ? config.key_length_swa : config.key_length;
    const std::uint32_t value_size = sliding ? config.value_length_swa : config.value_length;
    if (head_size != value_size) {
        throw std::runtime_error("Gemma 4 K/V head widths must match");
    }

    ggml_tensor * frequency_factors =
        sliding ? nullptr : model.text_tensor("rope_freqs.weight");

    auto * current = rms_norm(
        ctx, input, layer_tensor(model, layer, "attn_norm.weight"),
        config.layer_norm_epsilon);
    auto * query = ggml_mul_mat(
        ctx, layer_tensor(model, layer, "attn_q.weight"), current);
    query = ggml_reshape_3d(
        ctx, query, head_size, config.head_count, query_count);
    query = rms_norm(
        ctx, query, layer_tensor(model, layer, "attn_q_norm.weight"),
        config.layer_norm_epsilon);
    query = ggml_rope_ext(
        ctx, query, positions, frequency_factors,
        sliding ? config.rope_dimension_count_swa : config.rope_dimension_count,
        GGML_ROPE_TYPE_NEOX, 0,
        sliding ? config.rope_freq_base_swa : config.rope_freq_base,
        1.0F, 0.0F, 1.0F, 0.0F, 0.0F);

    const auto stored_layers = config.block_count - config.shared_kv_layers;
    ggml_tensor * key = nullptr;
    ggml_tensor * value = nullptr;
    if (layer < stored_layers) {
        key = ggml_mul_mat(
            ctx, layer_tensor(model, layer, "attn_k.weight"), current);
        value = ggml_mul_mat(
            ctx, layer_tensor(model, layer, "attn_v.weight"), current);
        key = ggml_reshape_3d(
            ctx, key, head_size, config.head_count_kv, query_count);
        value = ggml_reshape_3d(
            ctx, value, value_size, config.head_count_kv, query_count);
        key = rms_norm(
            ctx, key, layer_tensor(model, layer, "attn_k_norm.weight"),
            config.layer_norm_epsilon);
        value = ggml_rms_norm(ctx, value, config.layer_norm_epsilon);
        key = ggml_rope_ext(
            ctx, key, positions, frequency_factors,
            sliding ? config.rope_dimension_count_swa : config.rope_dimension_count,
            GGML_ROPE_TYPE_NEOX, 0,
            sliding ? config.rope_freq_base_swa : config.rope_freq_base,
            1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
        auto * key_copy = cache_current_kv(
            ctx, key, cache.key(layer), config.key_width(layer), query_count, cache_start);
        auto * value_copy = cache_current_kv(
            ctx, value, cache.value(layer), config.value_width(layer), query_count, cache_start);
        ggml_build_forward_expand(graph, query);
        ggml_build_forward_expand(graph, value_copy);
        ggml_build_forward_expand(graph, key_copy);
    }
    key = ggml_view_2d(
        ctx, cache.key(layer), config.key_width(layer), cache_count,
        cache.key(layer)->nb[1], 0);
    value = ggml_view_2d(
        ctx, cache.value(layer), config.value_width(layer), cache_count,
        cache.value(layer)->nb[1], 0);
    key = ggml_reshape_3d(
        ctx, key, head_size, config.head_count_kv, cache_count);
    value = ggml_reshape_3d(
        ctx, value, value_size, config.head_count_kv, cache_count);
    current = build_attention(
        ctx, query, key, value, sliding ? sliding_mask : full_mask);
    current = ggml_mul_mat(
        ctx, layer_tensor(model, layer, "attn_output.weight"), current);
    current = rms_norm(
        ctx, current, layer_tensor(model, layer, "post_attention_norm.weight"),
        config.layer_norm_epsilon);
    auto * attention_output = ggml_add(ctx, current, input);

    current = rms_norm(
        ctx, attention_output, layer_tensor(model, layer, "ffn_norm.weight"),
        config.layer_norm_epsilon);
    auto * up = ggml_mul_mat(
        ctx, layer_tensor(model, layer, "ffn_up.weight"), current);
    auto * gate = ggml_mul_mat(
        ctx, layer_tensor(model, layer, "ffn_gate.weight"), current);
    current = ggml_geglu_split(ctx, gate, up);
    current = ggml_mul_mat(
        ctx, layer_tensor(model, layer, "ffn_down.weight"), current);
    current = rms_norm(
        ctx, current, layer_tensor(model, layer, "post_ffw_norm.weight"),
        config.layer_norm_epsilon);
    current = ggml_add(ctx, current, attention_output);

    auto * residual = current;
    current = ggml_mul_mat(
        ctx, layer_tensor(model, layer, "inp_gate.weight"), current);
    current = ggml_gelu(ctx, current);
    const std::size_t slice_bytes =
        static_cast<std::size_t>(config.per_layer_embedding_length) *
        query_count * sizeof(float);
    auto * layer_input = ggml_view_2d(
        ctx, per_layer_inputs, config.per_layer_embedding_length, query_count,
        config.per_layer_embedding_length * sizeof(float), layer * slice_bytes);
    current = ggml_mul(ctx, current, layer_input);
    current = ggml_mul_mat(
        ctx, layer_tensor(model, layer, "proj.weight"), current);
    current = rms_norm(
        ctx, current, layer_tensor(model, layer, "post_norm.weight"),
        config.layer_norm_epsilon);
    current = ggml_add(ctx, residual, current);
    if (auto * scale = model.text_tensor(
            layer_name(layer, "layer_output_scale.weight"))) {
        current = ggml_mul(ctx, current, scale);
    }
    return current;
}

} // namespace

struct PrefillState::Impl {
    ~Impl() {
        if (logits_buffer != nullptr) ggml_backend_buffer_free(logits_buffer);
        if (logits_ctx != nullptr) ggml_free(logits_ctx);
    }

    std::unique_ptr<StateCache> cache;
    BackendContext * backend = nullptr;
    BackendTiming backend_timing;
    std::size_t graph_node_count = 0;
    ggml_context * logits_ctx = nullptr;
    ggml_backend_buffer_t logits_buffer = nullptr;
    ggml_tensor * logits = nullptr;
};

PrefillState::~PrefillState() = default;
PrefillState::PrefillState(PrefillState &&) noexcept = default;
PrefillState & PrefillState::operator=(PrefillState &&) noexcept = default;
PrefillState::PrefillState(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

std::size_t PrefillState::prefix_length() const noexcept {
    return impl_->cache->prefix_length();
}

StateCache & PrefillState::cache() noexcept { return *impl_->cache; }
const StateCache & PrefillState::cache() const noexcept { return *impl_->cache; }
ggml_tensor * PrefillState::logits() const noexcept { return impl_->logits; }

std::vector<float> PrefillState::download_logits() const {
    std::vector<float> values(ggml_nelements(impl_->logits));
    impl_->backend->tensor_get_timed(
        impl_->logits, values.data(), 0, values.size() * sizeof(float),
        impl_->backend_timing);
    return values;
}

BackendTiming PrefillState::backend_timing() const noexcept {
    return impl_->backend_timing;
}

std::size_t PrefillState::graph_node_count() const noexcept {
    return impl_->graph_node_count;
}

PrefillEngine::PrefillEngine(ModelBundle & model, BackendContext & backend)
    : model_(model), backend_(backend) {}

PrefillState PrefillEngine::prefill(
    const std::vector<TokenId> & tokens_before_image,
    const VisualTokens * visual_tokens,
    const std::vector<TokenId> & tokens_after_image) const {
    const auto & config = model_.text_config();
    if (visual_tokens != nullptr &&
        visual_tokens->embedding_length() != config.embedding_length) {
        throw std::runtime_error("visual-token width does not match text model");
    }
    const std::size_t visual_count =
        visual_tokens == nullptr ? 0 : visual_tokens->token_count();
    const std::size_t token_count =
        tokens_before_image.size() + visual_count + tokens_after_image.size();
    if (token_count == 0 || token_count > config.context_length) {
        throw std::runtime_error("prefill exceeds model context");
    }

    auto result = std::make_unique<PrefillState::Impl>();
    result->backend = &backend_;
    result->cache = std::make_unique<StateCache>(
        config, token_count, backend_);

    const std::size_t context_size =
        max_graph_nodes * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(max_graph_nodes, false);
    std::vector<std::uint8_t> context_memory(context_size);
    ggml_init_params params{context_memory.size(), context_memory.data(), true};
    using ContextPointer = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    ContextPointer ctx(ggml_init(params), ggml_free);
    if (!ctx) throw std::runtime_error("failed to create prefill graph context");
    auto * graph = ggml_new_graph_custom(ctx.get(), max_graph_nodes, false);

    std::vector<TokenId> all_token_ids;
    all_token_ids.reserve(token_count);
    all_token_ids.insert(
        all_token_ids.end(), tokens_before_image.begin(), tokens_before_image.end());
    all_token_ids.insert(all_token_ids.end(), visual_count, 0);
    all_token_ids.insert(
        all_token_ids.end(), tokens_after_image.begin(), tokens_after_image.end());
    auto * per_layer_ids = ggml_new_tensor_1d(
        ctx.get(), GGML_TYPE_I32, token_count);
    ggml_set_name(per_layer_ids, "prefill_token_ids");
    ggml_set_input(per_layer_ids);

    auto * token_embedding = require_text_tensor(model_, "token_embd.weight");
    ggml_tensor * input = nullptr;
    ggml_tensor * before_ids = nullptr;
    ggml_tensor * after_ids = nullptr;
    const float embedding_scale = std::sqrt(config.embedding_length);
    if (!tokens_before_image.empty()) {
        before_ids = ggml_new_tensor_1d(
            ctx.get(), GGML_TYPE_I32, tokens_before_image.size());
        ggml_set_input(before_ids);
        auto * segment = ggml_scale(
            ctx.get(), ggml_get_rows(ctx.get(), token_embedding, before_ids),
            embedding_scale);
        input = append_tokens(ctx.get(), input, segment);
    }
    if (visual_tokens != nullptr) {
        input = append_tokens(ctx.get(), input, visual_tokens->tensor());
    }
    if (!tokens_after_image.empty()) {
        after_ids = ggml_new_tensor_1d(
            ctx.get(), GGML_TYPE_I32, tokens_after_image.size());
        ggml_set_input(after_ids);
        auto * segment = ggml_scale(
            ctx.get(), ggml_get_rows(ctx.get(), token_embedding, after_ids),
            embedding_scale);
        input = append_tokens(ctx.get(), input, segment);
    }
    if (input == nullptr) throw std::runtime_error("prefill has no input embeddings");

    auto * positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, token_count);
    ggml_set_input(positions);
    auto * full_mask = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F32, token_count, token_count);
    auto * sliding_mask = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F32, token_count, token_count);
    ggml_set_input(full_mask);
    ggml_set_input(sliding_mask);

    auto * per_layer = ggml_get_rows(
        ctx.get(), require_text_tensor(model_, "per_layer_token_embd.weight"),
        per_layer_ids);
    per_layer = ggml_reshape_3d(
        ctx.get(), per_layer, config.per_layer_embedding_length,
        config.block_count, token_count);
    per_layer = ggml_scale(
        ctx.get(), per_layer, std::sqrt(config.per_layer_embedding_length));
    auto * projected = ggml_mul_mat(
        ctx.get(), require_text_tensor(model_, "per_layer_model_proj.weight"), input);
    projected = ggml_scale(
        ctx.get(), projected, 1.0F / std::sqrt(config.embedding_length));
    projected = ggml_reshape_3d(
        ctx.get(), projected, config.per_layer_embedding_length,
        config.block_count, token_count);
    projected = rms_norm(
        ctx.get(), projected,
        require_text_tensor(model_, "per_layer_proj_norm.weight"),
        config.layer_norm_epsilon);
    per_layer = ggml_scale(
        ctx.get(), ggml_add(ctx.get(), projected, per_layer),
        1.0F / std::sqrt(2.0F));
    per_layer = ggml_cont(
        ctx.get(), ggml_permute(ctx.get(), per_layer, 0, 2, 1, 3));

    for (std::uint32_t layer = 0; layer < config.block_count; ++layer) {
        input = build_layer(
            ctx.get(), graph, model_, *result->cache, config, layer, token_count,
            0, token_count,
            positions, full_mask, sliding_mask, per_layer, input);
    }
    input = rms_norm(
        ctx.get(), input, require_text_tensor(model_, "output_norm.weight"),
        config.layer_norm_epsilon);
    input = ggml_view_2d(
        ctx.get(), input, config.embedding_length, 1, input->nb[1],
        (token_count - 1) * input->nb[1]);
    auto * logits = ggml_mul_mat(ctx.get(), token_embedding, input);
    if (config.final_logit_softcap != 0.0F) {
        logits = ggml_scale(
            ctx.get(), logits, 1.0F / config.final_logit_softcap);
        logits = ggml_tanh(ctx.get(), logits);
        logits = ggml_scale(ctx.get(), logits, config.final_logit_softcap);
    }
    ggml_set_output(logits);
    ggml_build_forward_expand(graph, logits);

    using AllocatorPointer = std::unique_ptr<
        std::remove_pointer_t<ggml_gallocr_t>, decltype(&ggml_gallocr_free)>;
    AllocatorPointer allocator(
        ggml_gallocr_new(backend_.buffer_type()), ggml_gallocr_free);
    if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
        throw std::runtime_error("failed to allocate prefill graph on selected backend");
    }

    result->backend->tensor_set_timed(
        per_layer_ids, all_token_ids.data(), 0, ggml_nbytes(per_layer_ids),
        result->backend_timing);
    if (before_ids != nullptr) {
        result->backend->tensor_set_timed(
            before_ids, tokens_before_image.data(), 0, ggml_nbytes(before_ids),
            result->backend_timing);
    }
    if (after_ids != nullptr) {
        result->backend->tensor_set_timed(
            after_ids, tokens_after_image.data(), 0, ggml_nbytes(after_ids),
            result->backend_timing);
    }
    std::vector<std::int32_t> position_values(token_count);
    for (std::size_t index = 0; index < token_count; ++index) {
        position_values[index] = static_cast<std::int32_t>(index);
    }
    result->backend->tensor_set_timed(
        positions, position_values.data(), 0, ggml_nbytes(positions),
        result->backend_timing);
    const auto full_mask_values = causal_mask(0, token_count, token_count, 0);
    const auto sliding_mask_values = causal_mask(
        0, token_count, token_count, config.sliding_window);
    result->backend->tensor_set_timed(
        full_mask, full_mask_values.data(), 0, ggml_nbytes(full_mask),
        result->backend_timing);
    result->backend->tensor_set_timed(
        sliding_mask, sliding_mask_values.data(), 0, ggml_nbytes(sliding_mask),
        result->backend_timing);

    const auto status = ggml_backend_graph_compute(backend_.backend(), graph);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(
            "prefill graph compute failed: " + std::string(ggml_status_to_string(status)));
    }
    result->backend->synchronize(result->backend_timing);
    result->graph_node_count = ggml_graph_n_nodes(graph);

    ggml_init_params logits_params{3 * ggml_tensor_overhead(), nullptr, true};
    result->logits_ctx = ggml_init(logits_params);
    if (result->logits_ctx == nullptr) {
        throw std::runtime_error("failed to create persistent logits context");
    }
    result->logits = ggml_new_tensor_1d(
        result->logits_ctx, GGML_TYPE_F32, config.vocabulary_size);
    result->logits_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        result->logits_ctx, backend_.buffer_type());
    if (result->logits_buffer == nullptr) {
        throw std::runtime_error("failed to allocate persistent Prefill logits");
    }
    result->backend->tensor_copy_timed(
        logits, result->logits, result->backend_timing);
    result->backend->synchronize(result->backend_timing);
    result->cache->freeze_prefix(token_count);
    return PrefillState(std::move(result));
}

} // namespace branchscore
