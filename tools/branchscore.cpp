#include "branchscore/backend_context.hpp"
#include "branchscore/image_preprocessor.hpp"
#include "branchscore/model_loader.hpp"
#include "branchscore/vision_encoder.hpp"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

double gib(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

void print_devices() {
    const auto devices = branchscore::BackendContext::available_devices();
    if (devices.empty()) {
        std::cout << "No ggml backend devices found\n";
        return;
    }

    for (const auto & device : devices) {
        std::cout << device.name << "\tfamily=" << device.family
                  << "\ttype=" << device.type;
        if (device.memory_total != 0) {
            std::cout << "\tmemory=" << std::fixed << std::setprecision(2)
                      << gib(device.memory_free) << "/" << gib(device.memory_total)
                      << " GiB free";
        }
        std::cout << "\t" << device.description << '\n';
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        std::string selector = "auto";
        std::string model_path;
        std::string mmproj_path;
        std::string image_path;
        std::string vision_dump_path;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--list-backends") {
                print_devices();
                return 0;
            }
            if (arg == "--backend" && i + 1 < argc) {
                selector = argv[++i];
                continue;
            }
            if (arg == "--model" && i + 1 < argc) {
                model_path = argv[++i];
                continue;
            }
            if (arg == "--mmproj" && i + 1 < argc) {
                mmproj_path = argv[++i];
                continue;
            }
            if (arg == "--image" && i + 1 < argc) {
                image_path = argv[++i];
                continue;
            }
            if (arg == "--vision-dump" && i + 1 < argc) {
                vision_dump_path = argv[++i];
                continue;
            }
            if (arg == "--help") {
                std::cout
                    << "Usage: branchscore [--list-backends] [--backend NAME]\n"
                    << "                   [--model FILE --mmproj FILE [--image FILE]]\n"
                    << "                   [--vision-dump FILE]\n";
                return 0;
            }
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }

        branchscore::BackendContext backend(selector);
        std::cout << "Initialized " << backend.device().name
                  << " (" << backend.device().family << ", "
                  << backend.device().type << ")\n";

        if (model_path.empty() != mmproj_path.empty()) {
            throw std::runtime_error("--model and --mmproj must be supplied together");
        }
        if (!image_path.empty() && model_path.empty()) {
            throw std::runtime_error("--image requires --model and --mmproj");
        }
        if (!vision_dump_path.empty() && image_path.empty()) {
            throw std::runtime_error("--vision-dump requires --image");
        }
        if (!model_path.empty()) {
            auto model = branchscore::ModelLoader::load(model_path, mmproj_path, backend);
            const auto & text = model.text_config();
            const auto & vision = model.vision_config();
            std::cout << "Loaded " << text.name << ": " << text.block_count
                      << " text blocks, width " << text.embedding_length
                      << ", vocab " << text.vocabulary_size << ", "
                      << model.text_tensor_count() << " tensors ("
                      << std::fixed << std::setprecision(2)
                      << gib(model.text_weight_bytes()) << " GiB)\n";
            std::cout << "Loaded " << vision.projector_type << ": "
                      << vision.block_count << " vision blocks, width "
                      << vision.embedding_length << " -> "
                      << vision.projection_length << ", "
                      << model.vision_tensor_count() << " tensors ("
                      << gib(model.vision_weight_bytes()) << " GiB)\n";
            if (!image_path.empty()) {
                const auto image = branchscore::ImagePreprocessor::load(image_path, vision);
                std::cout << "Prepared image: " << image.width << "x" << image.height
                          << ", " << image.patch_count(vision) << " patches, "
                          << image.visual_token_count(vision) << " visual tokens\n";
                const auto started = std::chrono::steady_clock::now();
                const auto embeddings =
                    branchscore::VisionEncoder(model, backend).encode(image);
                const auto elapsed = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started);
                std::cout << "Encoded image: " << embeddings.token_count << "x"
                          << embeddings.embedding_length << " embeddings in "
                          << std::setprecision(2) << elapsed.count() << " ms\n";
                if (!vision_dump_path.empty()) {
                    std::ofstream dump(vision_dump_path, std::ios::binary);
                    if (!dump) {
                        throw std::runtime_error(
                            "failed to open vision dump '" + vision_dump_path + "'");
                    }
                    const std::int32_t header[] = {
                        static_cast<std::int32_t>(embeddings.token_count),
                        static_cast<std::int32_t>(embeddings.embedding_length),
                    };
                    dump.write(reinterpret_cast<const char *>(header), sizeof(header));
                    dump.write(
                        reinterpret_cast<const char *>(embeddings.values.data()),
                        static_cast<std::streamsize>(
                            embeddings.values.size() * sizeof(float)));
                    if (!dump) throw std::runtime_error("failed to write vision dump");
                }
            }
        }
        backend.synchronize();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "branchscore: " << error.what() << '\n';
        return 1;
    }
}
