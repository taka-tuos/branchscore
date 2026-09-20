#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace branchscore {

struct DecisionOption {
    std::string id;
    std::string description;
};

enum class ReasoningPolicy {
    DirectAnswerDisabled,
};

struct PromptPolicy {
    ReasoningPolicy reasoning = ReasoningPolicy::DirectAnswerDisabled;
};

const char * reasoning_policy_name(ReasoningPolicy policy) noexcept;

struct DecisionRequest {
    std::string state;
    std::string question;
    std::vector<DecisionOption> options;
    std::optional<std::string> image_path;
    PromptPolicy prompt_policy;
    std::optional<std::string> chat_template_file;
    bool capture_vision_debug = false;
};

struct PromptFormatInfo {
    std::string renderer_id;
    std::string model_family;
    std::string effective_source;
    PromptPolicy prompt_policy;
    std::optional<std::string> requested_template_file;
    bool requested_template_applied = false;
    bool gguf_chat_template_present = false;
    bool gguf_chat_template_used = false;
};

struct TimingInfo {
    double prompt_rendering_ms = 0.0;
    double tokenization_ms = 0.0;
    double image_preprocessing_ms = 0.0;
    double vision_ms = 0.0;
    double vision_backend_copy_ms = 0.0;
    double vision_synchronization_ms = 0.0;
    std::size_t vision_graph_node_count = 0;
    double prefill_ms = 0.0;
    double prefill_backend_copy_ms = 0.0;
    double prefill_synchronization_ms = 0.0;
    std::size_t prefill_graph_node_count = 0;
    std::vector<double> option_scoring_ms;
    double option_backend_copy_ms = 0.0;
    double option_synchronization_ms = 0.0;
    std::size_t option_graph_node_count = 0;
    double score_total_ms = 0.0;
    double normalization_ms = 0.0;
    double request_total_ms = 0.0;
};

struct OptionScore {
    std::size_t input_index = 0;
    std::string option_id;
    std::size_t token_count = 0;
    double sum_logprob = 0.0;
    double mean_logprob = 0.0;
    double relative_probability = 0.0;
    double elapsed_ms = 0.0;
    double backend_copy_ms = 0.0;
    double synchronization_ms = 0.0;
    std::size_t graph_node_count = 0;
    std::vector<std::int32_t> token_ids;
    std::vector<float> token_logprobs;
};

struct VisionDebugInfo {
    std::size_t token_count = 0;
    std::size_t embedding_length = 0;
    std::vector<float> values;
};

struct DecisionResult {
    std::vector<OptionScore> option_scores;
    std::size_t selected_index = 0;
    std::string selected_id;
    bool exact_tie = false;
    std::string scoring_basis = "sum_logprob";
    bool terminator_scored = false;
    PromptFormatInfo prompt_format;
    std::string rendered_prompt_identity;
    std::vector<std::int32_t> rendered_prefix_token_ids;
    std::optional<VisionDebugInfo> vision_debug;
    TimingInfo timings;
};

} // namespace branchscore
