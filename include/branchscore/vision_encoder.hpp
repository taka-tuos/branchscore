#pragma once

#include "branchscore/backend_context.hpp"
#include "branchscore/image_preprocessor.hpp"
#include "branchscore/model_loader.hpp"

#include <cstddef>
#include <vector>

namespace branchscore {

struct VisionEmbeddings {
    std::size_t token_count = 0;
    std::size_t embedding_length = 0;
    std::vector<float> values;
};

class VisionEncoder {
public:
    VisionEncoder(ModelBundle & model, BackendContext & backend);
    VisionEmbeddings encode(const PreparedImage & image) const;

private:
    ModelBundle & model_;
    BackendContext & backend_;
};

} // namespace branchscore
