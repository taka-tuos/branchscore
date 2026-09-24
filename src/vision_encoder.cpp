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
    ggml_tensor * input,
    bool use_flash_attention) {
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

    if (use_flash_attention) {
        value = ggml_permute(ctx, value, 0, 2, 1, 3);
        key = ggml_cast(ctx, key, GGML_TYPE_F16);
        value = ggml_cast(ctx, value, GGML_TYPE_F16);
        auto * attended = ggml_flash_attn_ext(
            ctx, query, key, value, nullptr, 1.0F, 0.0F, 0.0F);
        ggml_prec_set_acc(attended, GGML_PREC_F32);
        attended = ggml_reshape_2d(
            ctx, attended, attended->ne[0] * attended->ne[1],
            attended->ne[2] * attended->ne[3]);
        return clipped_mm(
            ctx, model, require_tensor(model, prefix + "attn_out.weight"), attended);
    }

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
    ggml_tensor * input,
    bool use_flash_attention) {
    const auto prefix = "v.blk." + std::to_string(layer) + ".";
    auto * current = rms_norm(
        ctx, input, require_tensor(model, prefix + "ln1.weight"),
        config.layer_norm_epsilon);
    current = attention(
        ctx, model, config, layer, position_count, pos_x, pos_y, current,
        use_flash_attention);
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

struct VisionGraph {
    std::vector<std::uint8_t> context_memory;
    ggml_context * context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * pos_x = nullptr;
    ggml_tensor * pos_y = nullptr;
    ggml_tensor * output = nullptr;

    ~VisionGraph() {
        if (context != nullptr) ggml_free(context);
    }
};

std::unique_ptr<VisionGraph> build_vision_graph(
    ModelBundle & model,
    const PreparedImage & image,
    bool use_flash_attention) {
    const auto & config = model.vision_config();
    const auto position_count = image.patch_count(config);
    const auto output_tokens = image.visual_token_count(config);
    const auto patches_x = image.width / config.patch_size;
    const auto patches_y = image.height / config.patch_size;

    auto result = std::make_unique<VisionGraph>();
    const std::size_t context_size =
        max_graph_nodes * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(max_graph_nodes, false);
    result->context_memory.resize(context_size);
    ggml_init_params params{
        result->context_memory.size(), result->context_memory.data(), true,
    };
    result->context = ggml_init(params);
    if (result->context == nullptr) {
        throw std::runtime_error("failed to create vision graph context");
    }
    result->graph = ggml_new_graph_custom(result->context, max_graph_nodes, false);

    result->input = ggml_new_tensor_4d(
        result->context, GGML_TYPE_F32, image.width, image.height, 3, 1);
    ggml_set_name(result->input, "vision_input");
    ggml_set_input(result->input);
    auto * current = ggml_scale_bias(result->context, result->input, 2.0F, -1.0F);
    current = ggml_conv_2d(
        result->context, require_tensor(model, "v.patch_embd.weight"), current,
        config.patch_size, config.patch_size, 0, 0, 1, 1);
    current = ggml_reshape_3d(
        result->context, current, position_count, config.embedding_length, 1);
    current = ggml_cont(result->context, ggml_transpose(result->context, current));

    result->pos_x = ggml_new_tensor_1d(
        result->context, GGML_TYPE_I32, position_count);
    result->pos_y = ggml_new_tensor_1d(
        result->context, GGML_TYPE_I32, position_count);
    ggml_set_name(result->pos_x, "vision_pos_x");
    ggml_set_name(result->pos_y, "vision_pos_y");
    ggml_set_input(result->pos_x);
    ggml_set_input(result->pos_y);

    auto * positions = require_tensor(model, "v.position_embd.weight");
    const auto position_table_size = positions->ne[1];
    const auto position_row_bytes =
        ggml_row_size(positions->type, config.embedding_length);
    auto * table_x = ggml_view_2d(
        result->context, positions, config.embedding_length, position_table_size,
        position_row_bytes, 0);
    auto * table_y = ggml_view_2d(
        result->context, positions, config.embedding_length, position_table_size,
        position_row_bytes, position_table_size * position_row_bytes);
    current = ggml_add(
        result->context, current,
        ggml_get_rows(result->context, table_x, result->pos_x));
    current = ggml_add(
        result->context, current,
        ggml_get_rows(result->context, table_y, result->pos_y));
    current = ggml_reshape_2d(
        result->context, current, config.embedding_length, position_count);

    for (std::uint32_t layer = 0; layer < config.block_count; ++layer) {
        current = transformer_layer(
            result->context, model, config, layer, position_count,
            result->pos_x, result->pos_y, current, use_flash_attention);
    }

    current = ggml_cont_4d(
        result->context, ggml_transpose(result->context, current), patches_x, patches_y,
        config.embedding_length, 1);
    current = ggml_pool_2d(
        result->context, current, GGML_OP_POOL_AVG, config.merge_size, config.merge_size,
        config.merge_size, config.merge_size, 0, 0);
    current = ggml_reshape_3d(
        result->context, current, output_tokens, config.embedding_length, 1);
    current = ggml_cont(result->context, ggml_transpose(result->context, current));
    current = ggml_scale(result->context, current, std::sqrt(config.embedding_length));
    current = ggml_rms_norm(result->context, current, config.layer_norm_epsilon);
    current = clipped_mm(
        result->context, model,
        require_tensor(model, "mm.input_projection.weight"), current);
    ggml_set_name(current, "vision_embeddings");
    ggml_set_output(current);
    ggml_build_forward_expand(result->graph, current);
    result->output = current;
    return result;
}

bool supports_flash_attention(
    const BackendContext & backend,
    ggml_cgraph * graph) {
    bool found_flash_attention = false;
    for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
        auto * node = ggml_graph_node(graph, index);
        if (node->op != GGML_OP_FLASH_ATTN_EXT) continue;
        found_flash_attention = true;
        if (!ggml_backend_supports_op(backend.backend(), node)) return false;
    }
    return found_flash_attention;
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
    VisionAttentionPath attention_path = VisionAttentionPath::Standard;
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

VisionAttentionPath VisualTokens::attention_path() const noexcept {
    return impl_->attention_path;
}

const char * vision_attention_path_name(const VisionAttentionPath path) noexcept {
    switch (path) {
        case VisionAttentionPath::Standard: return "standard";
        case VisionAttentionPath::Flash:    return "flash";
    }
    return "unknown";
}

VisionEncoder::VisionEncoder(ModelBundle & model, BackendContext & backend)
    : model_(model), backend_(backend) {}

VisualTokens VisionEncoder::encode(const PreparedImage & image) const {
    const auto & config = model_.vision_config();
    if (image.pixels.size() !=
        static_cast<std::size_t>(image.width) * image.height * 3) {
        throw std::runtime_error("prepared image buffer has an unexpected size");
    }
    const auto output_tokens = image.visual_token_count(config);
    auto graph = build_vision_graph(model_, image, true);
    VisionAttentionPath attention_path = VisionAttentionPath::Flash;
    if (!supports_flash_attention(backend_, graph->graph)) {
        graph = build_vision_graph(model_, image, false);
        attention_path = VisionAttentionPath::Standard;
    }

    using AllocatorPointer = std::unique_ptr<
        std::remove_pointer_t<ggml_gallocr_t>, decltype(&ggml_gallocr_free)>;
    AllocatorPointer allocator(
        ggml_gallocr_new(backend_.buffer_type()), ggml_gallocr_free);
    if (!allocator) throw std::runtime_error("failed to create vision graph allocator");
    if (!ggml_gallocr_alloc_graph(allocator.get(), graph->graph)) {
        throw std::runtime_error(
            "failed to allocate vision graph on selected backend (attention_path=" +
            std::string(vision_attention_path_name(attention_path)) + ")");
    }

    const auto position_count = image.patch_count(config);
    const auto patches_x = image.width / config.patch_size;
    const auto patches_y = image.height / config.patch_size;
    std::vector<std::int32_t> x_positions(position_count);
    std::vector<std::int32_t> y_positions(position_count);
    for (std::size_t index = 0; index < position_count; ++index) {
        x_positions[index] = static_cast<std::int32_t>(index % patches_x);
        y_positions[index] = static_cast<std::int32_t>(index / patches_x);
    }
    BackendTiming backend_timing;
    backend_.tensor_set_timed(
        graph->input, image.pixels.data(), 0, ggml_nbytes(graph->input), backend_timing);
    backend_.tensor_set_timed(
        graph->pos_x, x_positions.data(), 0, ggml_nbytes(graph->pos_x), backend_timing);
    backend_.tensor_set_timed(
        graph->pos_y, y_positions.data(), 0, ggml_nbytes(graph->pos_y), backend_timing);
    const auto status = ggml_backend_graph_compute(backend_.backend(), graph->graph);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(
            "vision graph compute failed (attention_path=" +
            std::string(vision_attention_path_name(attention_path)) + "): " +
            std::string(ggml_status_to_string(status)));
    }
    backend_.synchronize(backend_timing);

    const auto result_size = output_tokens * config.projection_length;
    if (graph->output->type != GGML_TYPE_F32 ||
        static_cast<std::size_t>(ggml_nelements(graph->output)) != result_size) {
        throw std::runtime_error("vision graph output has an unexpected shape or type");
    }

    auto result = std::make_unique<VisualTokens::Impl>();
    result->token_count = output_tokens;
    result->embedding_length = config.projection_length;
    result->backend = &backend_;
    result->backend_timing = backend_timing;
    result->graph_node_count = ggml_graph_n_nodes(graph->graph);
    result->attention_path = attention_path;
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
    backend_.tensor_copy_timed(graph->output, result->tensor, result->backend_timing);
    backend_.synchronize(result->backend_timing);
    return VisualTokens(std::move(result));
}

} // namespace branchscore
