#pragma once

#include "branchscore/backend_context.hpp"
#include "branchscore/image_preprocessor.hpp"
#include "branchscore/model_loader.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace branchscore {

class VisualTokens {
public:
    ~VisualTokens();
    VisualTokens(VisualTokens &&) noexcept;
    VisualTokens & operator=(VisualTokens &&) noexcept;

    VisualTokens(const VisualTokens &) = delete;
    VisualTokens & operator=(const VisualTokens &) = delete;

    std::size_t token_count() const noexcept;
    std::size_t embedding_length() const noexcept;
    ggml_tensor * tensor() const noexcept;
    std::vector<float> download() const;
    BackendTiming backend_timing() const noexcept;
    std::size_t graph_node_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    explicit VisualTokens(std::unique_ptr<Impl> impl);
    friend class VisionEncoder;
};

class VisionEncoder {
public:
    VisionEncoder(ModelBundle & model, BackendContext & backend);
    VisualTokens encode(const PreparedImage & image) const;

private:
    ModelBundle & model_;
    BackendContext & backend_;
};

} // namespace branchscore
