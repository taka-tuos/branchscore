#include "branchscore/chat_template.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace branchscore {

ChatTemplate::ChatTemplate(std::string source, std::string origin)
    : source_(std::move(source)), origin_(std::move(origin)) {
    if (source_.empty()) throw std::runtime_error("chat template is empty");
    for (const char * marker : {
             "bos_token", "<|turn>", "<turn|>", "add_generation_prompt"}) {
        if (source_.find(marker) == std::string::npos) {
            throw std::runtime_error(
                "chat template '" + origin_ + "' is not a supported Gemma 4 template; "
                "missing marker " + marker);
        }
    }
}

ChatTemplate ChatTemplate::from_source(std::string source, std::string origin) {
    return ChatTemplate(std::move(source), std::move(origin));
}

ChatTemplate ChatTemplate::from_file(const std::string & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("failed to open chat template file '" + path + "'");
    std::ostringstream source;
    source << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw std::runtime_error("failed to read chat template file '" + path + "'");
    }
    return from_source(source.str(), path);
}

const std::string & ChatTemplate::source() const noexcept { return source_; }
const std::string & ChatTemplate::origin() const noexcept { return origin_; }

std::string ChatTemplate::render(
    const ChatPrompt & prompt, bool add_generation_prompt) const {
    if (prompt.system.empty()) throw std::runtime_error("system prompt must not be empty");
    if (prompt.user.empty()) throw std::runtime_error("user prompt must not be empty");

    // The Phase 2 request shape intentionally exercises only the plain-text
    // and image content branches of the native Gemma 4 template. The source
    // is retained and validated above so a future full Jinja renderer can be
    // inserted behind this same boundary without changing callers.
    std::string result;
    result.reserve(prompt.system.size() + prompt.user.size() + 96);
    result += "<bos>";
    result += "<|turn>system\n";
    result += prompt.system;
    result += "<turn|>\n<|turn>user\n";
    if (prompt.image) result += "<|image|>\n";
    result += prompt.user;
    result += "<turn|>\n";
    if (add_generation_prompt) result += "<|turn>model\n";
    return result;
}

} // namespace branchscore
