#include "branchscore/image_preprocessor.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool close(float actual, float expected) {
    return std::abs(actual - expected) < 1e-6F;
}

branchscore::VisionModelConfig test_config() {
    branchscore::VisionModelConfig config;
    config.patch_size = 2;
    config.merge_size = 3;
    config.image_min_pixels = 70 * 36;
    config.image_max_pixels = 1120 * 36;
    config.image_mean = {0.5F, 0.25F, 0.0F};
    config.image_std = {0.5F, 0.25F, 1.0F};
    return config;
}

} // namespace

int main() {
    const auto config = test_config();
    const std::uint8_t pixel[] = {255, 128, 0};
    const auto small = branchscore::ImagePreprocessor::preprocess_rgb(
        pixel, 1, 1, config);

    bool valid = true;
    valid &= small.width == 54 && small.height == 54;
    valid &= small.patch_count(config) == 729;
    valid &= small.visual_token_count(config) == 81;
    const std::size_t plane = static_cast<std::size_t>(small.width) * small.height;
    valid &= close(small.pixels[0], 1.0F);
    valid &= close(small.pixels[plane], (128.0F / 255.0F - 0.25F) / 0.25F);
    valid &= close(small.pixels[2 * plane], 0.0F);

    std::vector<std::uint8_t> large(1000U * 500U * 3U, 64);
    const auto reduced = branchscore::ImagePreprocessor::preprocess_rgb(
        large.data(), 1000, 500, config);
    valid &= reduced.width == 282 && reduced.height == 138;
    valid &= reduced.visual_token_count(config) <= 1120;

    if (!valid) std::cerr << "image preprocessing result differs from expected values\n";
    return valid ? 0 : 1;
}
