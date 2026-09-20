#pragma once

#include "branchscore/decision.hpp"

#include <optional>
#include <string>

namespace branchscore {

struct RenderedPrompt {
    std::string text;
    PromptFormatInfo format;
    std::string identity;
};

class Gemma4PromptRenderer {
public:
    static const char * renderer_id() noexcept;

    RenderedPrompt render(
        const std::string & state,
        const std::string & question,
        bool has_image,
        const PromptPolicy & policy,
        const std::optional<std::string> & requested_template_file,
        bool gguf_chat_template_present) const;
};

} // namespace branchscore
