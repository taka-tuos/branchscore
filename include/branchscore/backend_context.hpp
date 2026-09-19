#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <string>
#include <vector>

namespace branchscore {

struct BackendDevice {
    std::string name;
    std::string family;
    std::string description;
    std::string type;
    std::size_t memory_free = 0;
    std::size_t memory_total = 0;
};

class BackendContext {
public:
    explicit BackendContext(const std::string & selector);
    ~BackendContext();

    BackendContext(const BackendContext &) = delete;
    BackendContext & operator=(const BackendContext &) = delete;

    BackendContext(BackendContext && other) noexcept;
    BackendContext & operator=(BackendContext && other) noexcept;

    static std::vector<BackendDevice> available_devices();

    ggml_backend_t backend() const noexcept;
    ggml_backend_buffer_type_t buffer_type() const noexcept;
    const BackendDevice & device() const noexcept;
    void synchronize() const;

private:
    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_type_t buffer_type_ = nullptr;
    BackendDevice device_;
};

} // namespace branchscore

