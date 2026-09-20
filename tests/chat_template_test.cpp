#include "branchscore/chat_template.hpp"

#include <iostream>
#include <stdexcept>

int main() {
    try {
        const auto templ = branchscore::ChatTemplate::from_source(
            "{{ bos_token }} <|turn> add_generation_prompt <turn|>", "test");
        const auto rendered = templ.render(
            {"Use the state.", "State: ready", true}, true);
        const std::string expected =
            "<bos><|turn>system\nUse the state.<turn|>\n<|turn>user\n"
            "<|image|>\nState: ready<turn|>\n<|turn>model\n";
        if (rendered != expected) throw std::runtime_error("rendered chat template differs");
        std::cout << "chat template origin=" << templ.origin() << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
