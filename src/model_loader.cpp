#include "branchscore/model_loader.hpp"

#include "gguf.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace branchscore {
namespace {

constexpr std::size_t copy_chunk_size = 16U * 1024U * 1024U;

class LoadedWeights {
public:
    LoadedWeights() = default;
    ~LoadedWeights() {
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
        if (weights_ctx_ != nullptr) {
            ggml_free(weights_ctx_);
        }
        if (source_ctx_ != nullptr) {
            ggml_free(source_ctx_);
        }
        if (gguf_ != nullptr) {
            gguf_free(gguf_);
        }
    }

    LoadedWeights(const LoadedWeights &) = delete;
    LoadedWeights & operator=(const LoadedWeights &) = delete;

    LoadedWeights(LoadedWeights && other) noexcept {
        swap(other);
    }

    LoadedWeights & operator=(LoadedWeights && other) noexcept {
        if (this != &other) {
            LoadedWeights temporary(std::move(other));
            swap(temporary);
        }
        return *this;
    }

    void swap(LoadedWeights & other) noexcept {
        std::swap(gguf_, other.gguf_);
        std::swap(source_ctx_, other.source_ctx_);
        std::swap(weights_ctx_, other.weights_ctx_);
        std::swap(buffer_, other.buffer_);
        tensors_.swap(other.tensors_);
        std::swap(weight_bytes_, other.weight_bytes_);
    }

    gguf_context * gguf() const noexcept { return gguf_; }

    ggml_tensor * tensor(const std::string & name) const noexcept {
        const auto found = tensors_.find(name);
        return found == tensors_.end() ? nullptr : found->second;
    }

    std::size_t tensor_count() const noexcept { return tensors_.size(); }
    std::size_t weight_bytes() const noexcept { return weight_bytes_; }

    static LoadedWeights load(
        const std::string & path,
        BackendContext & backend,
        const std::function<bool(const std::string &)> & include) {
        LoadedWeights result;
        gguf_init_params file_params{/* no_alloc = */ true, /* ctx = */ &result.source_ctx_};
        result.gguf_ = gguf_init_from_file(path.c_str(), file_params);
        if (result.gguf_ == nullptr || result.source_ctx_ == nullptr) {
            throw std::runtime_error("failed to read GGUF metadata from '" + path + "'");
        }

        std::vector<ggml_tensor *> sources;
        for (ggml_tensor * source = ggml_get_first_tensor(result.source_ctx_);
             source != nullptr;
             source = ggml_get_next_tensor(result.source_ctx_, source)) {
            if (include(source->name)) {
                sources.push_back(source);
            }
        }
        if (sources.empty()) {
            throw std::runtime_error("no selected tensors in '" + path + "'");
        }

        const std::size_t context_size =
            (sources.size() + 1U) * ggml_tensor_overhead() + ggml_graph_overhead();
        ggml_init_params weights_params{
            /* mem_size = */ context_size,
            /* mem_buffer = */ nullptr,
            /* no_alloc = */ true,
        };
        result.weights_ctx_ = ggml_init(weights_params);
        if (result.weights_ctx_ == nullptr) {
            throw std::runtime_error("failed to create tensor metadata context for '" + path + "'");
        }

        for (auto * source : sources) {
            auto * destination = ggml_dup_tensor(result.weights_ctx_, source);
            ggml_set_name(destination, source->name);
            result.tensors_.emplace(source->name, destination);
            result.weight_bytes_ += ggml_nbytes(destination);
        }

        result.buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(
            result.weights_ctx_, backend.buffer_type());
        if (result.buffer_ == nullptr) {
            throw std::runtime_error(
                "failed to allocate " + std::to_string(result.weight_bytes_) +
                " weight bytes on backend '" + backend.device().name + "'");
        }
        ggml_backend_buffer_set_usage(result.buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        std::ifstream input(path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("failed to open GGUF data file '" + path + "'");
        }
        std::vector<char> copy_buffer(copy_chunk_size);
        for (auto * source : sources) {
            const auto tensor_id = gguf_find_tensor(result.gguf_, source->name);
            if (tensor_id < 0) {
                throw std::runtime_error("GGUF tensor index missing for '" + std::string(source->name) + "'");
            }

            const auto file_offset = gguf_get_data_offset(result.gguf_) +
                                     gguf_get_tensor_offset(result.gguf_, tensor_id);
            input.seekg(static_cast<std::streamoff>(file_offset), std::ios::beg);
            if (!input) {
                throw std::runtime_error("failed to seek to tensor '" + std::string(source->name) + "'");
            }

            auto * destination = result.tensors_.at(source->name);
            const std::size_t tensor_bytes = ggml_nbytes(destination);
            for (std::size_t offset = 0; offset < tensor_bytes;) {
                const std::size_t size = std::min(copy_buffer.size(), tensor_bytes - offset);
                input.read(copy_buffer.data(), static_cast<std::streamsize>(size));
                if (input.gcount() != static_cast<std::streamsize>(size)) {
                    throw std::runtime_error("short read for tensor '" + std::string(source->name) + "'");
                }
                ggml_backend_tensor_set(destination, copy_buffer.data(), offset, size);
                offset += size;
            }
        }
        backend.synchronize();
        return result;
    }

private:
    gguf_context * gguf_ = nullptr;
    ggml_context * source_ctx_ = nullptr;
    ggml_context * weights_ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors_;
    std::size_t weight_bytes_ = 0;
};

int64_t require_key(const gguf_context * gguf, const std::string & key) {
    const auto id = gguf_find_key(gguf, key.c_str());
    if (id < 0) {
        throw std::runtime_error("required GGUF key is missing: " + key);
    }
    return id;
}

std::string string_value(const gguf_context * gguf, const std::string & key) {
    const auto id = require_key(gguf, key);
    if (gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) {
        throw std::runtime_error("GGUF key has wrong type: " + key);
    }
    return gguf_get_val_str(gguf, id);
}

std::uint32_t u32_value(const gguf_context * gguf, const std::string & key) {
    const auto id = require_key(gguf, key);
    if (gguf_get_kv_type(gguf, id) != GGUF_TYPE_UINT32) {
        throw std::runtime_error("GGUF key has wrong type: " + key);
    }
    return gguf_get_val_u32(gguf, id);
}

float f32_value(const gguf_context * gguf, const std::string & key) {
    const auto id = require_key(gguf, key);
    if (gguf_get_kv_type(gguf, id) != GGUF_TYPE_FLOAT32) {
        throw std::runtime_error("GGUF key has wrong type: " + key);
    }
    return gguf_get_val_f32(gguf, id);
}

std::uint32_t optional_u32_value(
    const gguf_context * gguf,
    const std::string & key,
    std::uint32_t fallback) {
    const auto id = gguf_find_key(gguf, key.c_str());
    if (id < 0) return fallback;
    if (gguf_get_kv_type(gguf, id) != GGUF_TYPE_UINT32) {
        throw std::runtime_error("GGUF key has wrong type: " + key);
    }
    return gguf_get_val_u32(gguf, id);
}

std::array<float, 3> f32_triplet(const gguf_context * gguf, const std::string & key) {
    const auto id = require_key(gguf, key);
    if (gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(gguf, id) != GGUF_TYPE_FLOAT32 ||
        gguf_get_arr_n(gguf, id) < 3) {
        throw std::runtime_error("GGUF key is not a float32 triplet: " + key);
    }
    const auto * values = static_cast<const float *>(gguf_get_arr_data(gguf, id));
    return {values[0], values[1], values[2]};
}

std::vector<std::uint32_t> u32_values(const gguf_context * gguf, const std::string & key) {
    const auto id = require_key(gguf, key);
    const auto checked = [&key](std::int64_t value) {
        if (value < 0 || static_cast<std::uint64_t>(value) >
                             std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("GGUF value is outside uint32 range: " + key);
        }
        return static_cast<std::uint32_t>(value);
    };

    switch (gguf_get_kv_type(gguf, id)) {
        case GGUF_TYPE_UINT32: return {gguf_get_val_u32(gguf, id)};
        case GGUF_TYPE_INT32:  return {checked(gguf_get_val_i32(gguf, id))};
        case GGUF_TYPE_UINT64: {
            const auto value = gguf_get_val_u64(gguf, id);
            if (value > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("GGUF value is outside uint32 range: " + key);
            }
            return {static_cast<std::uint32_t>(value)};
        }
        case GGUF_TYPE_INT64: return {checked(gguf_get_val_i64(gguf, id))};
        default: break;
    }
    if (gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY) {
        throw std::runtime_error("GGUF key is not an integer value or array: " + key);
    }

    const auto count = gguf_get_arr_n(gguf, id);
    const auto * data = gguf_get_arr_data(gguf, id);
    std::vector<std::uint32_t> result;
    result.reserve(count);
    switch (gguf_get_arr_type(gguf, id)) {
        case GGUF_TYPE_UINT32: {
            const auto * values = static_cast<const std::uint32_t *>(data);
            result.assign(values, values + count);
            break;
        }
        case GGUF_TYPE_INT32: {
            const auto * values = static_cast<const std::int32_t *>(data);
            for (std::size_t i = 0; i < count; ++i) result.push_back(checked(values[i]));
            break;
        }
        case GGUF_TYPE_UINT64: {
            const auto * values = static_cast<const std::uint64_t *>(data);
            for (std::size_t i = 0; i < count; ++i) {
                if (values[i] > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::runtime_error("GGUF value is outside uint32 range: " + key);
                }
                result.push_back(static_cast<std::uint32_t>(values[i]));
            }
            break;
        }
        case GGUF_TYPE_INT64: {
            const auto * values = static_cast<const std::int64_t *>(data);
            for (std::size_t i = 0; i < count; ++i) result.push_back(checked(values[i]));
            break;
        }
        default:
            throw std::runtime_error("GGUF array is not integer-valued: " + key);
    }
    return result;
}

std::vector<bool> bool_values(const gguf_context * gguf, const std::string & key) {
    const auto id = require_key(gguf, key);
    if (gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY) {
        throw std::runtime_error("GGUF key is not a bool array: " + key);
    }
    const auto count = gguf_get_arr_n(gguf, id);
    std::vector<bool> result;
    result.reserve(count);
    const auto * data = gguf_get_arr_data(gguf, id);
    if (gguf_get_arr_type(gguf, id) == GGUF_TYPE_BOOL ||
        gguf_get_arr_type(gguf, id) == GGUF_TYPE_UINT8 ||
        gguf_get_arr_type(gguf, id) == GGUF_TYPE_INT8) {
        const auto * values = static_cast<const std::uint8_t *>(data);
        for (std::size_t i = 0; i < count; ++i) result.push_back(values[i] != 0);
    } else if (gguf_get_arr_type(gguf, id) == GGUF_TYPE_UINT32) {
        const auto * values = static_cast<const std::uint32_t *>(data);
        for (std::size_t i = 0; i < count; ++i) result.push_back(values[i] != 0);
    } else if (gguf_get_arr_type(gguf, id) == GGUF_TYPE_INT32) {
        const auto * values = static_cast<const std::int32_t *>(data);
        for (std::size_t i = 0; i < count; ++i) result.push_back(values[i] != 0);
    } else {
        throw std::runtime_error("GGUF array is not bool-compatible: " + key);
    }
    return result;
}

void require_tensor(const LoadedWeights & weights, const std::string & name) {
    if (weights.tensor(name) == nullptr) {
        throw std::runtime_error("required tensor is missing: " + name);
    }
}

std::string block_tensor(std::uint32_t block, const std::string & suffix) {
    return "blk." + std::to_string(block) + "." + suffix;
}

std::string vision_block_tensor(std::uint32_t block, const std::string & suffix) {
    return "v.blk." + std::to_string(block) + "." + suffix;
}

TextModelConfig read_text_config(const gguf_context * gguf) {
    TextModelConfig config;
    config.architecture = string_value(gguf, "general.architecture");
    config.name = string_value(gguf, "general.name");
    if (config.architecture != "gemma4") {
        throw std::runtime_error("expected general.architecture=gemma4, got '" + config.architecture + "'");
    }

    config.context_length = u32_value(gguf, "gemma4.context_length");
    config.embedding_length = u32_value(gguf, "gemma4.embedding_length");
    config.block_count = u32_value(gguf, "gemma4.block_count");
    config.head_count = u32_value(gguf, "gemma4.attention.head_count");
    config.head_count_kv = u32_value(gguf, "gemma4.attention.head_count_kv");
    config.key_length = u32_value(gguf, "gemma4.attention.key_length");
    config.value_length = u32_value(gguf, "gemma4.attention.value_length");
    config.key_length_swa = u32_value(gguf, "gemma4.attention.key_length_swa");
    config.value_length_swa = u32_value(gguf, "gemma4.attention.value_length_swa");
    config.sliding_window = u32_value(gguf, "gemma4.attention.sliding_window");
    config.shared_kv_layers = u32_value(gguf, "gemma4.attention.shared_kv_layers");
    config.per_layer_embedding_length =
        u32_value(gguf, "gemma4.embedding_length_per_layer_input");
    config.final_logit_softcap = f32_value(gguf, "gemma4.final_logit_softcapping");
    config.feed_forward_lengths = u32_values(gguf, "gemma4.feed_forward_length");
    config.sliding_window_pattern =
        bool_values(gguf, "gemma4.attention.sliding_window_pattern");
    config.vocabulary_size = gguf_get_arr_n(
        gguf, require_key(gguf, "tokenizer.ggml.tokens"));

    if (config.block_count != 35 && config.block_count != 42) {
        throw std::runtime_error(
            "only Gemma 4 E2B/E4B are supported; block_count=" +
            std::to_string(config.block_count));
    }
    if (config.sliding_window_pattern.size() != config.block_count) {
        throw std::runtime_error("sliding-window pattern length does not match block count");
    }
    if (config.feed_forward_lengths.size() != 1 &&
        config.feed_forward_lengths.size() != config.block_count) {
        throw std::runtime_error("feed-forward length metadata has unexpected size");
    }
    return config;
}

VisionModelConfig read_vision_config(const gguf_context * gguf) {
    VisionModelConfig config;
    config.projector_type = string_value(gguf, "clip.vision.projector_type");
    if (config.projector_type != "gemma4v") {
        throw std::runtime_error(
            "expected clip.vision.projector_type=gemma4v, got '" +
            config.projector_type + "'");
    }
    config.projection_length = u32_value(gguf, "clip.vision.projection_dim");
    config.image_size = u32_value(gguf, "clip.vision.image_size");
    config.patch_size = u32_value(gguf, "clip.vision.patch_size");
    config.embedding_length = u32_value(gguf, "clip.vision.embedding_length");
    config.feed_forward_length = u32_value(gguf, "clip.vision.feed_forward_length");
    config.block_count = u32_value(gguf, "clip.vision.block_count");
    config.head_count = u32_value(gguf, "clip.vision.attention.head_count");
    config.merge_size = optional_u32_value(
        gguf, "clip.vision.projector.scale_factor", 3);
    config.layer_norm_epsilon =
        f32_value(gguf, "clip.vision.attention.layer_norm_epsilon");
    config.image_mean = f32_triplet(gguf, "clip.vision.image_mean");
    config.image_std = f32_triplet(gguf, "clip.vision.image_std");

    if (config.merge_size == 0 || config.patch_size == 0) {
        throw std::runtime_error("Gemma 4 vision patch and merge sizes must be positive");
    }
    const std::uint64_t aligned_patch =
        static_cast<std::uint64_t>(config.patch_size) * config.merge_size;
    const std::uint64_t patch_area = aligned_patch * aligned_patch;
    const std::uint64_t min_pixels = 70U * patch_area;
    const std::uint64_t max_pixels = 1120U * patch_area;
    if (max_pixels > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Gemma 4 vision pixel limits exceed uint32 range");
    }
    config.image_min_pixels = static_cast<std::uint32_t>(min_pixels);
    config.image_max_pixels = static_cast<std::uint32_t>(max_pixels);
    for (std::size_t channel = 0; channel < 3; ++channel) {
        if (config.image_std[channel] == 0.0F) {
            throw std::runtime_error("Gemma 4 vision image std must be nonzero");
        }
    }
    return config;
}

void validate_text_tensors(const LoadedWeights & weights, const TextModelConfig & config) {
    for (const char * name : {
             "token_embd.weight",
             "output_norm.weight",
             "per_layer_token_embd.weight",
             "per_layer_model_proj.weight",
             "per_layer_proj_norm.weight"}) {
        require_tensor(weights, name);
    }

    const auto kv_layer_count = config.block_count - config.shared_kv_layers;
    for (std::uint32_t block = 0; block < config.block_count; ++block) {
        for (const char * suffix : {
                 "attn_norm.weight",
                 "attn_q.weight",
                 "attn_q_norm.weight",
                 "attn_output.weight",
                 "post_attention_norm.weight",
                 "ffn_norm.weight",
                 "ffn_gate.weight",
                 "ffn_up.weight",
                 "ffn_down.weight",
                 "post_ffw_norm.weight",
                 "inp_gate.weight",
                 "proj.weight",
                 "post_norm.weight"}) {
            require_tensor(weights, block_tensor(block, suffix));
        }
        if (block < kv_layer_count) {
            require_tensor(weights, block_tensor(block, "attn_k.weight"));
            require_tensor(weights, block_tensor(block, "attn_k_norm.weight"));
        }
    }
}

void validate_vision_tensors(const LoadedWeights & weights, const VisionModelConfig & config) {
    for (const char * name : {
             "v.patch_embd.weight",
             "v.position_embd.weight",
             "mm.input_projection.weight"}) {
        require_tensor(weights, name);
    }
    for (std::uint32_t block = 0; block < config.block_count; ++block) {
        for (const char * suffix : {
                 "ln1.weight",
                 "ln2.weight",
                 "attn_q.weight",
                 "attn_k.weight",
                 "attn_v.weight",
                 "attn_out.weight",
                 "attn_q_norm.weight",
                 "attn_k_norm.weight",
                 "attn_post_norm.weight",
                 "ffn_gate.weight",
                 "ffn_up.weight",
                 "ffn_down.weight",
                 "ffn_post_norm.weight"}) {
            require_tensor(weights, vision_block_tensor(block, suffix));
        }
    }
}

} // namespace

struct ModelBundle::Impl {
    TextModelConfig text_config;
    VisionModelConfig vision_config;
    LoadedWeights text_weights;
    LoadedWeights vision_weights;
};

ModelBundle::~ModelBundle() = default;
ModelBundle::ModelBundle(ModelBundle &&) noexcept = default;
ModelBundle & ModelBundle::operator=(ModelBundle &&) noexcept = default;

ModelBundle::ModelBundle(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

const TextModelConfig & ModelBundle::text_config() const noexcept {
    return impl_->text_config;
}

const VisionModelConfig & ModelBundle::vision_config() const noexcept {
    return impl_->vision_config;
}

ggml_tensor * ModelBundle::text_tensor(const std::string & name) const noexcept {
    return impl_->text_weights.tensor(name);
}

ggml_tensor * ModelBundle::vision_tensor(const std::string & name) const noexcept {
    return impl_->vision_weights.tensor(name);
}

std::size_t ModelBundle::text_tensor_count() const noexcept {
    return impl_->text_weights.tensor_count();
}

std::size_t ModelBundle::vision_tensor_count() const noexcept {
    return impl_->vision_weights.tensor_count();
}

std::size_t ModelBundle::text_weight_bytes() const noexcept {
    return impl_->text_weights.weight_bytes();
}

std::size_t ModelBundle::vision_weight_bytes() const noexcept {
    return impl_->vision_weights.weight_bytes();
}

ModelBundle ModelLoader::load(
    const std::string & text_model_path,
    const std::string & mmproj_path,
    BackendContext & backend) {
    auto impl = std::make_unique<ModelBundle::Impl>();
    impl->text_weights = LoadedWeights::load(
        text_model_path, backend, [](const std::string &) { return true; });
    impl->text_config = read_text_config(impl->text_weights.gguf());
    validate_text_tensors(impl->text_weights, impl->text_config);

    impl->vision_weights = LoadedWeights::load(
        mmproj_path, backend, [](const std::string & name) {
            return name.rfind("v.", 0) == 0 || name == "mm.input_projection.weight";
        });
    impl->vision_config = read_vision_config(impl->vision_weights.gguf());
    validate_vision_tensors(impl->vision_weights, impl->vision_config);

    if (impl->vision_config.projection_length != impl->text_config.embedding_length) {
        throw std::runtime_error("mmproj projection width does not match text embedding width");
    }
    return ModelBundle(std::move(impl));
}

} // namespace branchscore
