#include "branchscore/vision_encoder.hpp"

#include "ggml-alloc.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace branchscore {
namespace {

constexpr std::size_t max_graph_nodes = 4096;

ggml_tensor * require_tensor(ModelBundle & model, const std::string & name) {
    auto * tensor = model.vision_tensor(name);
    if (tensor == nullptr) throw std::runtime_error("missing vision tensor: " + name);
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

ggml_tensor * clipped_mm(
    ggml_context * ctx,
    ModelBundle & model,
    ggml_tensor * weight,
    ggml_tensor * input) {
    const std::string weight_name = weight->name;
    const std::string stem = weight_name.substr(0, weight_name.size() - 7);
    const float maximum = std::numeric_limits<float>::max();
    const float input_min = model.vision_scalar(stem + ".input_min", -maximum);
    const float input_max = model.vision_scalar(stem + ".input_max", maximum);
    const float output_min = model.vision_scalar(stem + ".output_min", -maximum);
    const float output_max = model.vision_scalar(stem + ".output_max", maximum);
    auto * clamped = ggml_clamp(ctx, input, input_min, input_max);
    return ggml_clamp(ctx, ggml_mul_mat(ctx, weight, clamped), output_min, output_max);
}

ggml_tensor * add_rope(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * pos_x,
    ggml_tensor * pos_y) {
    const auto dimensions = static_cast<int>(input->ne[0]);
    auto * result = ggml_rope_ext(
        ctx, input, pos_x, nullptr, dimensions / 2, GGML_ROPE_TYPE_NEOX,
        0, 100.0F, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
    result = ggml_rope_ext(
        ctx, result, pos_y, nullptr, dimensions / 2, GGML_ROPE_TYPE_NEOX,
        0, 100.0F, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
    return ggml_rope_set_offset(result, dimensions / 2);
}

ggml_tensor * attention(
    ggml_context * ctx,
    ModelBundle & model,
    const VisionModelConfig & config,
    std::uint32_t layer,
    std::size_t position_count,
    ggml_tensor * pos_x,
    ggml_tensor * pos_y,
    ggml_tensor * input) {
    const auto prefix = "v.blk." + std::to_string(layer) + ".";
    const auto head_size = config.embedding_length / config.head_count;

    auto * query = clipped_mm(
        ctx, model, require_tensor(model, prefix + "attn_q.weight"), input);
    auto * key = clipped_mm(
        ctx, model, require_tensor(model, prefix + "attn_k.weight"), input);
    auto * value = clipped_mm(
        ctx, model, require_tensor(model, prefix + "attn_v.weight"), input);
    query = ggml_reshape_4d(
        ctx, query, head_size, config.head_count, position_count, 1);
    key = ggml_reshape_4d(
        ctx, key, head_size, config.head_count, position_count, 1);
    value = ggml_reshape_4d(
        ctx, value, head_size, config.head_count, position_count, 1);

    query = rms_norm(
        ctx, query, require_tensor(model, prefix + "attn_q_norm.weight"),
        config.layer_norm_epsilon);
    key = rms_norm(
        ctx, key, require_tensor(model, prefix + "attn_k_norm.weight"),
        config.layer_norm_epsilon);
    value = ggml_rms_norm(ctx, value, config.layer_norm_epsilon);
    query = add_rope(ctx, query, pos_x, pos_y);
    key = add_rope(ctx, key, pos_x, pos_y);

    query = ggml_permute(ctx, query, 0, 2, 1, 3);
    key = ggml_permute(ctx, key, 0, 2, 1, 3);
    value = ggml_cont(ctx, ggml_permute(ctx, value, 1, 2, 0, 3));
    auto * scores = ggml_mul_mat(ctx, key, query);
    scores = ggml_soft_max_ext(ctx, scores, nullptr, 1.0F, 0.0F);
    auto * attended = ggml_mul_mat(ctx, value, scores);
    attended = ggml_permute(ctx, attended, 0, 2, 1, 3);
    attended = ggml_cont_2d(
        ctx, attended, attended->ne[0] * attended->ne[1],
        attended->ne[2] * attended->ne[3]);
    return clipped_mm(
        ctx, model, require_tensor(model, prefix + "attn_out.weight"), attended);
}

ggml_tensor * transformer_layer(
    ggml_context * ctx,
    ModelBundle & model,
    const VisionModelConfig & config,
    std::uint32_t layer,
    std::size_t position_count,
    ggml_tensor * pos_x,
    ggml_tensor * pos_y,
    ggml_tensor * input) {
    const auto prefix = "v.blk." + std::to_string(layer) + ".";
    auto * current = rms_norm(
        ctx, input, require_tensor(model, prefix + "ln1.weight"),
        config.layer_norm_epsilon);
    current = attention(
        ctx, model, config, layer, position_count, pos_x, pos_y, current);
    current = rms_norm(
        ctx, current, require_tensor(model, prefix + "attn_post_norm.weight"),
        config.layer_norm_epsilon);
    current = ggml_add(ctx, current, input);
    auto * residual = current;

    current = rms_norm(
        ctx, current, require_tensor(model, prefix + "ln2.weight"),
        config.layer_norm_epsilon);
    auto * up = clipped_mm(
        ctx, model, require_tensor(model, prefix + "ffn_up.weight"), current);
    auto * gate = clipped_mm(
        ctx, model, require_tensor(model, prefix + "ffn_gate.weight"), current);
    current = ggml_geglu_quick_split(ctx, gate, up);
    current = clipped_mm(
        ctx, model, require_tensor(model, prefix + "ffn_down.weight"), current);
    current = rms_norm(
        ctx, current, require_tensor(model, prefix + "ffn_post_norm.weight"),
        config.layer_norm_epsilon);
    return ggml_add(ctx, residual, current);
}

} // namespace

struct VisualTokens::Impl {
    ~Impl() {
        if (buffer != nullptr) ggml_backend_buffer_free(buffer);
        if (ctx != nullptr) ggml_free(ctx);
    }

    std::size_t token_count = 0;
    std::size_t embedding_length = 0;
    BackendContext * backend = nullptr;
    BackendTiming backend_timing;
    std::size_t graph_node_count = 0;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * tensor = nullptr;
};

VisualTokens::~VisualTokens() = default;
VisualTokens::VisualTokens(VisualTokens &&) noexcept = default;
VisualTokens & VisualTokens::operator=(VisualTokens &&) noexcept = default;
VisualTokens::VisualTokens(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

std::size_t VisualTokens::token_count() const noexcept {
    return impl_->token_count;
}

std::size_t VisualTokens::embedding_length() const noexcept {
    return impl_->embedding_length;
}

ggml_tensor * VisualTokens::tensor() const noexcept {
    return impl_->tensor;
}

std::vector<float> VisualTokens::download() const {
    std::vector<float> values(token_count() * embedding_length());
    impl_->backend->tensor_get_timed(
        tensor(), values.data(), 0, values.size() * sizeof(float), impl_->backend_timing);
    return values;
}

BackendTiming VisualTokens::backend_timing() const noexcept {
    return impl_->backend_timing;
}

std::size_t VisualTokens::graph_node_count() const noexcept {
    return impl_->graph_node_count;
}

VisionEncoder::VisionEncoder(ModelBundle & model, BackendContext & backend)
    : model_(model), backend_(backend) {}

VisualTokens VisionEncoder::encode(const PreparedImage & image) const {
    const auto & config = model_.vision_config();
    if (image.pixels.size() !=
        static_cast<std::size_t>(image.width) * image.height * 3) {
        throw std::runtime_error("prepared image buffer has an unexpected size");
    }
    const auto position_count = image.patch_count(config);
    const auto output_tokens = image.visual_token_count(config);
    const auto patches_x = image.width / config.patch_size;
    const auto patches_y = image.height / config.patch_size;

    const std::size_t context_size =
        max_graph_nodes * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(max_graph_nodes, false);
    std::vector<std::uint8_t> context_memory(context_size);
    ggml_init_params params{
        context_memory.size(), context_memory.data(), true,
    };
    using ContextPointer = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    ContextPointer ctx(ggml_init(params), ggml_free);
    if (!ctx) throw std::runtime_error("failed to create vision graph context");
    auto * graph = ggml_new_graph_custom(ctx.get(), max_graph_nodes, false);

    auto * input = ggml_new_tensor_4d(
        ctx.get(), GGML_TYPE_F32, image.width, image.height, 3, 1);
    ggml_set_name(input, "vision_input");
    ggml_set_input(input);
    auto * current = ggml_scale_bias(ctx.get(), input, 2.0F, -1.0F);
    current = ggml_conv_2d(
        ctx.get(), require_tensor(model_, "v.patch_embd.weight"), current,
        config.patch_size, config.patch_size, 0, 0, 1, 1);
    current = ggml_reshape_3d(
        ctx.get(), current, position_count, config.embedding_length, 1);
    current = ggml_cont(ctx.get(), ggml_transpose(ctx.get(), current));

    auto * pos_x = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, position_count);
    auto * pos_y = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, position_count);
    ggml_set_name(pos_x, "vision_pos_x");
    ggml_set_name(pos_y, "vision_pos_y");
    ggml_set_input(pos_x);
    ggml_set_input(pos_y);

    auto * positions = require_tensor(model_, "v.position_embd.weight");
    const auto position_table_size = positions->ne[1];
    const auto position_row_bytes =
        ggml_row_size(positions->type, config.embedding_length);
    auto * table_x = ggml_view_2d(
        ctx.get(), positions, config.embedding_length, position_table_size,
        position_row_bytes, 0);
    auto * table_y = ggml_view_2d(
        ctx.get(), positions, config.embedding_length, position_table_size,
        position_row_bytes, position_table_size * position_row_bytes);
    current = ggml_add(ctx.get(), current, ggml_get_rows(ctx.get(), table_x, pos_x));
    current = ggml_add(ctx.get(), current, ggml_get_rows(ctx.get(), table_y, pos_y));
    current = ggml_reshape_2d(
        ctx.get(), current, config.embedding_length, position_count);

    for (std::uint32_t layer = 0; layer < config.block_count; ++layer) {
        current = transformer_layer(
            ctx.get(), model_, config, layer, position_count, pos_x, pos_y, current);
    }

    current = ggml_cont_4d(
        ctx.get(), ggml_transpose(ctx.get(), current), patches_x, patches_y,
        config.embedding_length, 1);
    current = ggml_pool_2d(
        ctx.get(), current, GGML_OP_POOL_AVG, config.merge_size, config.merge_size,
        config.merge_size, config.merge_size, 0, 0);
    current = ggml_reshape_3d(
        ctx.get(), current, output_tokens, config.embedding_length, 1);
    current = ggml_cont(ctx.get(), ggml_transpose(ctx.get(), current));
    current = ggml_scale(ctx.get(), current, std::sqrt(config.embedding_length));
    current = ggml_rms_norm(ctx.get(), current, config.layer_norm_epsilon);
    current = clipped_mm(
        ctx.get(), model_, require_tensor(model_, "mm.input_projection.weight"), current);
    ggml_set_name(current, "vision_embeddings");
    ggml_set_output(current);
    ggml_build_forward_expand(graph, current);

    using AllocatorPointer = std::unique_ptr<
        std::remove_pointer_t<ggml_gallocr_t>, decltype(&ggml_gallocr_free)>;
    AllocatorPointer allocator(
        ggml_gallocr_new(backend_.buffer_type()), ggml_gallocr_free);
    if (!allocator) throw std::runtime_error("failed to create vision graph allocator");
    if (!ggml_gallocr_alloc_graph(allocator.get(), graph)) {
        throw std::runtime_error("failed to allocate vision graph on selected backend");
    }

    std::vector<std::int32_t> x_positions(position_count);
    std::vector<std::int32_t> y_positions(position_count);
    for (std::size_t index = 0; index < position_count; ++index) {
        x_positions[index] = static_cast<std::int32_t>(index % patches_x);
        y_positions[index] = static_cast<std::int32_t>(index / patches_x);
    }
    BackendTiming backend_timing;
    backend_.tensor_set_timed(
        input, image.pixels.data(), 0, ggml_nbytes(input), backend_timing);
    backend_.tensor_set_timed(
        pos_x, x_positions.data(), 0, ggml_nbytes(pos_x), backend_timing);
    backend_.tensor_set_timed(
        pos_y, y_positions.data(), 0, ggml_nbytes(pos_y), backend_timing);
    const auto status = ggml_backend_graph_compute(backend_.backend(), graph);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(
            "vision graph compute failed: " + std::string(ggml_status_to_string(status)));
    }
    backend_.synchronize(backend_timing);

    const auto result_size = output_tokens * config.projection_length;
    if (current->type != GGML_TYPE_F32 ||
        static_cast<std::size_t>(ggml_nelements(current)) != result_size) {
        throw std::runtime_error("vision graph output has an unexpected shape or type");
    }

    auto result = std::make_unique<VisualTokens::Impl>();
    result->token_count = output_tokens;
    result->embedding_length = config.projection_length;
    result->backend = &backend_;
    result->backend_timing = backend_timing;
    result->graph_node_count = ggml_graph_n_nodes(graph);
    ggml_init_params output_params{
        2 * ggml_tensor_overhead(), nullptr, true,
    };
    result->ctx = ggml_init(output_params);
    if (result->ctx == nullptr) {
        throw std::runtime_error("failed to create persistent visual-token context");
    }
    result->tensor = ggml_new_tensor_2d(
        result->ctx, GGML_TYPE_F32, config.projection_length, output_tokens);
    ggml_set_name(result->tensor, "visual_tokens");
    result->buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        result->ctx, backend_.buffer_type());
    if (result->buffer == nullptr) {
        throw std::runtime_error("failed to allocate persistent visual-token buffer");
    }
    backend_.tensor_copy_timed(current, result->tensor, result->backend_timing);
    backend_.synchronize(result->backend_timing);
    return VisualTokens(std::move(result));
}

} // namespace branchscore
