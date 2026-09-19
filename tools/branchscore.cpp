#include "branchscore/backend_context.hpp"
#include "branchscore/image_preprocessor.hpp"
#include "branchscore/model_loader.hpp"

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
            if (arg == "--help") {
                std::cout
                    << "Usage: branchscore [--list-backends] [--backend NAME]\n"
                    << "                   [--model FILE --mmproj FILE [--image FILE]]\n";
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
            }
        }
        backend.synchronize();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "branchscore: " << error.what() << '\n';
        return 1;
    }
}
