#pragma once

#include "branchscore/backend_context.hpp"

#include "ggml.h"

#include <cstddef>
#include <cstdint>
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
    std::vector<std::uint32_t> feed_forward_lengths;
    std::vector<bool> sliding_window_pattern;
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
