#include "branchscore/backend_context.hpp"

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
            if (arg == "--help") {
                std::cout << "Usage: branchscore [--list-backends] [--backend NAME]\n";
                return 0;
            }
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }

        branchscore::BackendContext backend(selector);
        std::cout << "Initialized " << backend.device().name
                  << " (" << backend.device().family << ", "
                  << backend.device().type << ")\n";
        backend.synchronize();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "branchscore: " << error.what() << '\n';
        return 1;
    }
}

