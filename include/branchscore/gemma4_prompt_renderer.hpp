#pragma once

#include "branchscore/decision.hpp"

#include <optional>
#include <string>
#include <vector>

namespace branchscore {

struct RenderedAnswerSlot {
    std::string label;
    std::size_t input_index = 0;
    std::string option_id;
};

struct RenderedPrompt {
    std::string text;
    PromptFormatInfo format;
    std::string identity;
    std::vector<RenderedAnswerSlot> answer_slots;
};

class Gemma4PromptRenderer {
public:
    static const char * renderer_id() noexcept;

    RenderedPrompt render(
        const std::string & state,
        const std::string & question,
        const std::vector<DecisionOption> & options,
        bool has_image,
        const PromptPolicy & policy,
        const std::optional<std::string> & requested_template_file,
        bool gguf_chat_template_present) const;
};

} // namespace branchscore
