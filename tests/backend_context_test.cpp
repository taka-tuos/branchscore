#include "branchscore/backend_context.hpp"

#include <algorithm>
#include <iostream>

int main() {
    const auto devices = branchscore::BackendContext::available_devices();
    if (devices.empty()) {
        std::cerr << "no backend devices reported\n";
        return 1;
    }

    const auto cpu = std::find_if(devices.begin(), devices.end(), [](const auto & device) {
        return device.type == "cpu";
    });
    if (cpu == devices.end()) {
        std::cerr << "CPU backend is missing\n";
        return 1;
    }

    branchscore::BackendContext context(cpu->name);
    if (context.backend() == nullptr || context.buffer_type() == nullptr ||
        context.device().name != cpu->name) {
        std::cerr << "CPU backend initialization returned invalid state\n";
        return 1;
    }
    context.synchronize();
}
