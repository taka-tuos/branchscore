#include "branchscore/backend_context.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace branchscore {
namespace {

std::once_flag load_backends_once;
using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point started) {
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

void load_backends() {
    std::call_once(load_backends_once, [] { ggml_backend_load_all(); });
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "integrated-gpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accelerator";
        case GGML_BACKEND_DEVICE_TYPE_META:  return "meta";
    }
    return "unknown";
}

BackendDevice describe(ggml_backend_dev_t device) {
    ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(device, &props);

    const auto reg = ggml_backend_dev_backend_reg(device);
    BackendDevice result;
    result.name = props.name != nullptr ? props.name : "";
    result.family = reg != nullptr ? ggml_backend_reg_name(reg) : "";
    result.description = props.description != nullptr ? props.description : "";
    result.type = device_type_name(props.type);
    result.memory_free = props.memory_free;
    result.memory_total = props.memory_total;
    return result;
}

bool matches(ggml_backend_dev_t device, const std::string & selector) {
    const auto info = describe(device);
    const auto wanted = lower(selector);
    return lower(info.name) == wanted || lower(info.family) == wanted;
}

ggml_backend_dev_t select_device(const std::string & selector) {
    const auto count = ggml_backend_dev_count();
    if (count == 0) {
        throw std::runtime_error("ggml reported no available backend devices");
    }

    if (lower(selector) == "auto") {
        for (std::size_t i = 0; i < count; ++i) {
            auto * device = ggml_backend_dev_get(i);
            const auto type = ggml_backend_dev_type(device);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                return device;
            }
        }
        return ggml_backend_dev_get(0);
    }

    for (std::size_t i = 0; i < count; ++i) {
        auto * device = ggml_backend_dev_get(i);
        if (matches(device, selector)) {
            return device;
        }
    }

    std::ostringstream message;
    message << "backend '" << selector << "' is unavailable; available devices:";
    for (std::size_t i = 0; i < count; ++i) {
        const auto info = describe(ggml_backend_dev_get(i));
        message << " " << info.name << " (" << info.family << ")";
    }
    throw std::runtime_error(message.str());
}

} // namespace

BackendContext::BackendContext(const std::string & selector) {
    load_backends();
    auto * selected = select_device(selector);
    device_ = describe(selected);
    buffer_type_ = ggml_backend_dev_buffer_type(selected);
    if (buffer_type_ == nullptr) {
        throw std::runtime_error("backend '" + device_.name + "' has no default buffer type");
    }

    backend_ = ggml_backend_dev_init(selected, nullptr);
    if (backend_ == nullptr) {
        buffer_type_ = nullptr;
        throw std::runtime_error("failed to initialize backend '" + device_.name + "'");
    }
}

BackendContext::~BackendContext() {
    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
    }
}

BackendContext::BackendContext(BackendContext && other) noexcept
    : backend_(std::exchange(other.backend_, nullptr)),
      buffer_type_(std::exchange(other.buffer_type_, nullptr)),
      device_(std::move(other.device_)) {}

BackendContext & BackendContext::operator=(BackendContext && other) noexcept {
    if (this != &other) {
        if (backend_ != nullptr) {
            ggml_backend_free(backend_);
        }
        backend_ = std::exchange(other.backend_, nullptr);
        buffer_type_ = std::exchange(other.buffer_type_, nullptr);
        device_ = std::move(other.device_);
    }
    return *this;
}

std::vector<BackendDevice> BackendContext::available_devices() {
    load_backends();
    std::vector<BackendDevice> result;
    result.reserve(ggml_backend_dev_count());
    for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        result.push_back(describe(ggml_backend_dev_get(i)));
    }
    return result;
}

ggml_backend_t BackendContext::backend() const noexcept {
    return backend_;
}

ggml_backend_buffer_type_t BackendContext::buffer_type() const noexcept {
    return buffer_type_;
}

const BackendDevice & BackendContext::device() const noexcept {
    return device_;
}

void BackendContext::synchronize() const {
    ggml_backend_synchronize(backend_);
}

void BackendContext::synchronize(BackendTiming & timing) const {
    const auto started = Clock::now();
    ggml_backend_synchronize(backend_);
    timing.synchronization_ms += elapsed_ms(started);
}

void BackendContext::tensor_set_timed(
    ggml_tensor * tensor,
    const void * data,
    const std::size_t offset,
    const std::size_t size,
    BackendTiming & timing) const {
    const auto started = Clock::now();
    ggml_backend_tensor_set(tensor, data, offset, size);
    timing.copy_ms += elapsed_ms(started);
}

void BackendContext::tensor_get_timed(
    const ggml_tensor * tensor,
    void * data,
    const std::size_t offset,
    const std::size_t size,
    BackendTiming & timing) const {
    const auto started = Clock::now();
    ggml_backend_tensor_get(tensor, data, offset, size);
    timing.copy_ms += elapsed_ms(started);
}

void BackendContext::tensor_copy_timed(
    const ggml_tensor * source,
    ggml_tensor * destination,
    BackendTiming & timing) const {
    const auto started = Clock::now();
    ggml_backend_tensor_copy(source, destination);
    timing.copy_ms += elapsed_ms(started);
}

} // namespace branchscore
