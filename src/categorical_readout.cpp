#include "branchscore/categorical_readout.hpp"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace branchscore {
namespace {

constexpr std::size_t max_graph_nodes = 128;

} // namespace

CategoricalLogits gather_categorical_logits(
    const PrefillState & state,
    const std::vector<TokenId> & answer_token_ids,
    BackendContext & backend) {
    if (answer_token_ids.empty()) {
        throw std::runtime_error("categorical readout requires at least one answer token");
    }
    if (state.logits() == nullptr) {
        throw std::runtime_error("categorical readout has no Prefill logits");
    }

    const auto context_size =
        max_graph_nodes * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(max_graph_nodes, false);
    std::vector<std::uint8_t> context_memory(context_size);
    ggml_init_params params{context_memory.size(), context_memory.data(), true};
    using ContextPointer = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    ContextPointer ctx(ggml_init(params), ggml_free);
    if (!ctx) throw std::runtime_error("failed to create categorical readout context");
    auto * graph = ggml_new_graph_custom(ctx.get(), max_graph_nodes, false);

    auto * logits_rows = ggml_reshape_2d(
        ctx.get(), state.logits(), 1, ggml_nelements(state.logits()));
    auto * token_ids = ggml_new_tensor_1d(
        ctx.get(), GGML_TYPE_I32, answer_token_ids.size());
    ggml_set_input(token_ids);
    auto * gathered = ggml_get_rows(ctx.get(), logits_rows, token_ids);
    ggml_set_output(gathered);
    ggml_build_forward_expand(graph, gathered);

    using AllocatorPointer = std::unique_ptr<
        std::remove_pointer_t<ggml_gallocr_t>, decltype(&ggml_gallocr_free)>;
    AllocatorPointer allocator(
        ggml_gallocr_new(backend.buffer_type()), ggml_gallocr_free);
    if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
        throw std::runtime_error("failed to allocate categorical readout graph");
    }

    CategoricalLogits result;
    backend.tensor_set_timed(
        token_ids,
        answer_token_ids.data(),
        0,
        ggml_nbytes(token_ids),
        result.backend_timing);
    const auto status = ggml_backend_graph_compute(backend.backend(), graph);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(
            "categorical readout graph compute failed: " +
            std::string(ggml_status_to_string(status)));
    }
    backend.synchronize(result.backend_timing);

    result.raw_scores.resize(answer_token_ids.size());
    backend.tensor_get_timed(
        gathered,
        result.raw_scores.data(),
        0,
        result.raw_scores.size() * sizeof(float),
        result.backend_timing);
    result.graph_node_count = ggml_graph_n_nodes(graph);
    for (const auto value : result.raw_scores) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("categorical answer logits must be finite");
        }
    }
    return result;
}

CategoricalSummary summarize_categorical(const std::vector<float> & raw_scores) {
    if (raw_scores.empty()) {
        throw std::runtime_error("cannot normalize an empty categorical readout");
    }
    const auto maximum = std::max_element(raw_scores.begin(), raw_scores.end());
    if (!std::isfinite(*maximum)) {
        throw std::runtime_error("categorical answer logits must be finite");
    }

    double denominator = 0.0;
    for (const auto value : raw_scores) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("categorical answer logits must be finite");
        }
        denominator += std::exp(static_cast<double>(value - *maximum));
    }
    if (!std::isfinite(denominator) || denominator <= 0.0) {
        throw std::runtime_error("categorical softmax normalization failed");
    }

    CategoricalSummary result;
    result.relative_probabilities.reserve(raw_scores.size());
    result.selected_index = static_cast<std::size_t>(
        maximum - raw_scores.begin());
    for (const auto value : raw_scores) {
        result.relative_probabilities.push_back(
            std::exp(static_cast<double>(value - *maximum)) / denominator);
    }
    for (std::size_t index = 0; index < raw_scores.size(); ++index) {
        if (index != result.selected_index &&
            raw_scores[index] == raw_scores[result.selected_index]) {
            result.exact_tie = true;
            break;
        }
    }
    return result;
}

} // namespace branchscore
