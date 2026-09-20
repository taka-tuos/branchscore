#pragma once

#include <string>

namespace branchscore {

struct ChatPrompt {
    std::string system;
    std::string user;
    bool image = false;
};

class ChatTemplate {
public:
    static ChatTemplate from_source(std::string source, std::string origin);
    static ChatTemplate from_file(const std::string & path);

    const std::string & source() const noexcept;
    const std::string & origin() const noexcept;

    std::string render(const ChatPrompt & prompt, bool add_generation_prompt) const;

private:
    ChatTemplate(std::string source, std::string origin);

    std::string source_;
    std::string origin_;
};

} // namespace branchscore
