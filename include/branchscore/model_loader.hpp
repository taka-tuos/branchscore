#pragma once

#include "branchscore/backend_context.hpp"

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace branchscore {

struct TextModelConfig {
    std::string architecture;
    std::string name;
    std::uint32_t context_length = 0;
    std::uint32_t embedding_length = 0;
    std::uint32_t block_count = 0;
    std::uint32_t head_count = 0;
    std::uint32_t head_count_kv = 0;
    std::uint32_t key_length = 0;
    std::uint32_t value_length = 0;
    std::uint32_t key_length_swa = 0;
    std::uint32_t value_length_swa = 0;
    std::uint32_t sliding_window = 0;
    std::uint32_t shared_kv_layers = 0;
    std::uint32_t per_layer_embedding_length = 0;
    std::size_t vocabulary_size = 0;
    float final_logit_softcap = 0.0F;
    float layer_norm_epsilon = 0.0F;
    float rope_freq_base = 0.0F;
    float rope_freq_base_swa = 0.0F;
    std::uint32_t rope_dimension_count = 0;
    std::uint32_t rope_dimension_count_swa = 0;
    std::vector<std::uint32_t> feed_forward_lengths;
    std::vector<bool> sliding_window_pattern;

    bool is_sliding_window(std::uint32_t layer) const;
    std::uint32_t key_width(std::uint32_t layer) const;
    std::uint32_t value_width(std::uint32_t layer) const;
};

struct VisionModelConfig {
    std::string projector_type;
    std::uint32_t projection_length = 0;
    std::uint32_t image_size = 0;
    std::uint32_t patch_size = 0;
    std::uint32_t embedding_length = 0;
    std::uint32_t feed_forward_length = 0;
    std::uint32_t block_count = 0;
    std::uint32_t head_count = 0;
    std::uint32_t merge_size = 3;
    std::uint32_t image_min_pixels = 0;
    std::uint32_t image_max_pixels = 0;
    float layer_norm_epsilon = 0.0F;
    std::array<float, 3> image_mean{};
    std::array<float, 3> image_std{};
};

class ModelBundle {
public:
    ~ModelBundle();

    ModelBundle(const ModelBundle &) = delete;
    ModelBundle & operator=(const ModelBundle &) = delete;
    ModelBundle(ModelBundle &&) noexcept;
    ModelBundle & operator=(ModelBundle &&) noexcept;

    const TextModelConfig & text_config() const noexcept;
    const VisionModelConfig & vision_config() const noexcept;

    ggml_tensor * text_tensor(const std::string & name) const noexcept;
    ggml_tensor * vision_tensor(const std::string & name) const noexcept;
    float vision_scalar(const std::string & name, float fallback) const;

    std::size_t text_tensor_count() const noexcept;
    std::size_t vision_tensor_count() const noexcept;
    std::size_t text_weight_bytes() const noexcept;
    std::size_t vision_weight_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    explicit ModelBundle(std::unique_ptr<Impl> impl);
    friend class ModelLoader;
};

class ModelLoader {
public:
    static ModelBundle load(
        const std::string & text_model_path,
        const std::string & mmproj_path,
        BackendContext & backend);
};

} // namespace branchscore
