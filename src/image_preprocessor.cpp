#include "branchscore/image_preprocessor.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <climits>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace branchscore {
namespace {

struct ImageSize {
    int width;
    int height;
};

int checked_int(std::uint32_t value, const char * label) {
    if (value == 0 || value > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(label) + " is outside the supported range");
    }
    return static_cast<int>(value);
}

ImageSize target_size(int width, int height, const VisionModelConfig & config) {
    const int align = checked_int(config.patch_size * config.merge_size, "image alignment");
    if (config.image_min_pixels == 0 ||
        config.image_max_pixels < config.image_min_pixels) {
        throw std::runtime_error("invalid Gemma 4 vision pixel limits");
    }

    const auto round_factor = [align](double value) {
        return std::max(align, static_cast<int>(std::round(value / align)) * align);
    };
    const auto ceil_factor = [align](double value) {
        return static_cast<int>(std::ceil(value / align)) * align;
    };
    const auto floor_factor = [align](double value) {
        return std::max(align, static_cast<int>(std::floor(value / align)) * align);
    };

    int target_width = round_factor(width);
    int target_height = round_factor(height);
    const std::int64_t area =
        static_cast<std::int64_t>(target_width) * target_height;
    if (area > config.image_max_pixels) {
        const double scale = std::sqrt(
            static_cast<double>(width) * height / config.image_max_pixels);
        target_width = floor_factor(width / scale);
        target_height = floor_factor(height / scale);
    } else if (area < config.image_min_pixels) {
        const double scale = std::sqrt(
            static_cast<double>(config.image_min_pixels) /
            (static_cast<double>(width) * height));
        target_width = ceil_factor(width * scale);
        target_height = ceil_factor(height * scale);
    }
    return {target_width, target_height};
}

struct ResampleAxis {
    int kernel_size = 0;
    std::vector<int> bounds;
    std::vector<std::int32_t> weights;
};

ResampleAxis bicubic_axis(int input_size, int output_size) {
    constexpr int precision_bits = 22;
    constexpr double support_base = 2.0;
    const double scale = static_cast<double>(input_size) / output_size;
    const double filter_scale = std::max(1.0, scale);
    const double support = support_base * filter_scale;

    ResampleAxis axis;
    axis.kernel_size = static_cast<int>(std::ceil(support)) * 2 + 1;
    axis.bounds.resize(static_cast<std::size_t>(output_size) * 2);
    std::vector<double> raw(
        static_cast<std::size_t>(output_size) * axis.kernel_size, 0.0);

    const auto filter = [](double x) {
        constexpr double a = -0.5;
        x = std::abs(x);
        if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
        if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
        return 0.0;
    };

    for (int out = 0; out < output_size; ++out) {
        const double center = (out + 0.5) * scale;
        const int first = std::max(0, static_cast<int>(center - support + 0.5));
        const int end = std::min(input_size, static_cast<int>(center + support + 0.5));
        const int count = end - first;
        axis.bounds[2 * out] = first;
        axis.bounds[2 * out + 1] = count;

        double sum = 0.0;
        for (int index = 0; index < count; ++index) {
            const double weight = filter(
                (index + first - center + 0.5) / filter_scale);
            raw[static_cast<std::size_t>(out) * axis.kernel_size + index] = weight;
            sum += weight;
        }
        if (sum != 0.0) {
            for (int index = 0; index < count; ++index) {
                raw[static_cast<std::size_t>(out) * axis.kernel_size + index] /= sum;
            }
        }
    }

    const double fixed_scale = std::ldexp(1.0, precision_bits);
    axis.weights.reserve(raw.size());
    for (double value : raw) {
        const double rounded = value * fixed_scale + (value < 0.0 ? -0.5 : 0.5);
        axis.weights.push_back(static_cast<std::int32_t>(rounded));
    }
    return axis;
}

std::uint8_t clamp_u8(std::int64_t value) {
    return static_cast<std::uint8_t>(std::max<std::int64_t>(0, std::min<std::int64_t>(255, value)));
}

std::vector<std::uint8_t> resize_bicubic(
    const std::uint8_t * input,
    int input_width,
    int input_height,
    int output_width,
    int output_height) {
    constexpr int precision_bits = 22;
    if (input_width == output_width && input_height == output_height) {
        return {input, input + static_cast<std::size_t>(input_width) * input_height * 3};
    }

    const auto horizontal = bicubic_axis(input_width, output_width);
    const auto vertical = bicubic_axis(input_height, output_height);
    std::vector<std::uint8_t> intermediate(
        static_cast<std::size_t>(output_width) * input_height * 3);
    for (int y = 0; y < input_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            const int first = horizontal.bounds[2 * x];
            const int count = horizontal.bounds[2 * x + 1];
            for (int channel = 0; channel < 3; ++channel) {
                std::int64_t sum = 1LL << (precision_bits - 1);
                for (int index = 0; index < count; ++index) {
                    const auto source = input[
                        (static_cast<std::size_t>(y) * input_width + first + index) * 3 + channel];
                    sum += static_cast<std::int64_t>(source) * horizontal.weights[
                        static_cast<std::size_t>(x) * horizontal.kernel_size + index];
                }
                intermediate[
                    (static_cast<std::size_t>(y) * output_width + x) * 3 + channel] =
                    clamp_u8(sum >> precision_bits);
            }
        }
    }

    std::vector<std::uint8_t> output(
        static_cast<std::size_t>(output_width) * output_height * 3);
    for (int y = 0; y < output_height; ++y) {
        const int first = vertical.bounds[2 * y];
        const int count = vertical.bounds[2 * y + 1];
        for (int x = 0; x < output_width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                std::int64_t sum = 1LL << (precision_bits - 1);
                for (int index = 0; index < count; ++index) {
                    const auto source = intermediate[
                        (static_cast<std::size_t>(first + index) * output_width + x) * 3 + channel];
                    sum += static_cast<std::int64_t>(source) * vertical.weights[
                        static_cast<std::size_t>(y) * vertical.kernel_size + index];
                }
                output[(static_cast<std::size_t>(y) * output_width + x) * 3 + channel] =
                    clamp_u8(sum >> precision_bits);
            }
        }
    }
    return output;
}

} // namespace

std::size_t PreparedImage::patch_count(const VisionModelConfig & config) const {
    if (config.patch_size == 0 || width % config.patch_size != 0 ||
        height % config.patch_size != 0) {
        throw std::runtime_error("prepared image dimensions are not patch-aligned");
    }
    return static_cast<std::size_t>(width / config.patch_size) *
           (height / config.patch_size);
}

std::size_t PreparedImage::visual_token_count(const VisionModelConfig & config) const {
    if (config.merge_size == 0) {
        throw std::runtime_error("vision merge size must be positive");
    }
    const auto patches_x = width / config.patch_size;
    const auto patches_y = height / config.patch_size;
    if (patches_x % config.merge_size != 0 || patches_y % config.merge_size != 0) {
        throw std::runtime_error("prepared image patch grid is not merge-aligned");
    }
    return static_cast<std::size_t>(patches_x / config.merge_size) *
           (patches_y / config.merge_size);
}

PreparedImage ImagePreprocessor::load(
    const std::string & path,
    const VisionModelConfig & config) {
    int width = 0;
    int height = 0;
    int channels = 0;
    using ImagePointer = std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>;
    ImagePointer pixels(stbi_load(path.c_str(), &width, &height, &channels, 3), stbi_image_free);
    if (!pixels) {
        const char * reason = stbi_failure_reason();
        throw std::runtime_error(
            "failed to decode image '" + path + "': " +
            (reason == nullptr ? "unknown stb_image error" : reason));
    }
    return preprocess_rgb(
        pixels.get(), static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height), config);
}

EncodedImageInfo ImagePreprocessor::inspect_encoded(
    const std::uint8_t * encoded,
    const std::size_t encoded_size) {
    if (encoded == nullptr || encoded_size == 0) {
        throw ImageDecodeError("encoded image data must not be empty");
    }
    if (encoded_size > static_cast<std::size_t>(INT_MAX)) {
        throw ImageDecodeError("encoded image exceeds the decoder size limit");
    }
    int width = 0;
    int height = 0;
    int channels = 0;
    if (!stbi_info_from_memory(
            encoded, static_cast<int>(encoded_size), &width, &height, &channels)) {
        const char * reason = stbi_failure_reason();
        throw ImageDecodeError(
            std::string("failed to inspect encoded image: ") +
            (reason == nullptr ? "unknown stb_image error" : reason));
    }
    if (width <= 0 || height <= 0) {
        throw ImageDecodeError("encoded image has invalid dimensions");
    }
    return {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)};
}

PreparedImage ImagePreprocessor::load_encoded(
    const std::uint8_t * encoded,
    const std::size_t encoded_size,
    const VisionModelConfig & config) {
    const auto info = inspect_encoded(encoded, encoded_size);
    int width = 0;
    int height = 0;
    int channels = 0;
    using ImagePointer = std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>;
    ImagePointer pixels(
        stbi_load_from_memory(
            encoded, static_cast<int>(encoded_size), &width, &height, &channels, 3),
        stbi_image_free);
    if (!pixels) {
        const char * reason = stbi_failure_reason();
        throw ImageDecodeError(
            std::string("failed to decode encoded image: ") +
            (reason == nullptr ? "unknown stb_image error" : reason));
    }
    if (width != static_cast<int>(info.width) || height != static_cast<int>(info.height)) {
        throw ImageDecodeError("encoded image dimensions changed during decoding");
    }
    return preprocess_rgb(
        pixels.get(), static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height), config);
}

PreparedImage ImagePreprocessor::preprocess_rgb(
    const std::uint8_t * rgb,
    std::uint32_t width,
    std::uint32_t height,
    const VisionModelConfig & config) {
    if (rgb == nullptr) throw std::runtime_error("image RGB data is null");
    const int input_width = checked_int(width, "image width");
    const int input_height = checked_int(height, "image height");
    const auto target = target_size(input_width, input_height, config);
    const auto resized = resize_bicubic(
        rgb, input_width, input_height, target.width, target.height);

    PreparedImage result;
    result.width = static_cast<std::uint32_t>(target.width);
    result.height = static_cast<std::uint32_t>(target.height);
    const std::size_t pixel_count =
        static_cast<std::size_t>(result.width) * result.height;
    result.pixels.resize(pixel_count * 3);
    for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const float value = resized[pixel * 3 + channel] / 255.0F;
            result.pixels[channel * pixel_count + pixel] =
                (value - config.image_mean[channel]) / config.image_std[channel];
        }
    }
    return result;
}

} // namespace branchscore
