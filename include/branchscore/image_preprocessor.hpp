#pragma once

#include "branchscore/model_loader.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace branchscore {

struct PreparedImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float> pixels;

    std::size_t patch_count(const VisionModelConfig & config) const;
    std::size_t visual_token_count(const VisionModelConfig & config) const;
};

class ImagePreprocessor {
public:
    static PreparedImage load(
        const std::string & path,
        const VisionModelConfig & config);

    static PreparedImage preprocess_rgb(
        const std::uint8_t * rgb,
        std::uint32_t width,
        std::uint32_t height,
        const VisionModelConfig & config);
};

} // namespace branchscore
