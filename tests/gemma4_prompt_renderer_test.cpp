#include "branchscore/gemma4_prompt_renderer.hpp"

#include <iostream>
#include <stdexcept>

int main() {
    try {
        branchscore::Gemma4PromptRenderer renderer;
        const branchscore::PromptPolicy policy;
        const auto plain = renderer.render(
            "The service is healthy.",
            "Which action should be taken?",
            false,
            policy,
            std::nullopt,
            true);
        const std::string expected =
            "<bos><|turn>system\n"
            "Use the supplied state to answer the question. Return only the answer."
            "<turn|>\n<|turn>user\n"
            "State:\nThe service is healthy.\n\n"
            "Question:\nWhich action should be taken?"
            "<turn|>\n<|turn>model\n";
        if (plain.text != expected || plain.format.renderer_id != "gemma4-fixed-v1" ||
            plain.format.effective_source != "built-in" ||
            plain.format.requested_template_applied ||
            !plain.format.gguf_chat_template_present ||
            plain.format.gguf_chat_template_used ||
            plain.format.prompt_policy.reasoning !=
                branchscore::ReasoningPolicy::DirectAnswerDisabled ||
            plain.identity.find("gemma4-fixed-v1/sha256:") != 0) {
            throw std::runtime_error("text-only Gemma 4 renderer contract is incorrect");
        }

        const auto image = renderer.render(
            "white background",
            "What is visible?",
            true,
            policy,
            std::string("/does/not/exist.jinja"),
            false);
        const auto expected_image =
            "<bos><|turn>system\n"
            "Use the supplied state to answer the question. Return only the answer."
            "<turn|>\n<|turn>user\n<|image|>\n"
            "State:\nwhite background\n\nQuestion:\nWhat is visible?"
            "<turn|>\n<|turn>model\n";
        if (image.text != expected_image ||
            !image.format.requested_template_file ||
            *image.format.requested_template_file != "/does/not/exist.jinja" ||
            image.format.requested_template_applied ||
            image.format.gguf_chat_template_present ||
            image.format.gguf_chat_template_used ||
            image.identity == plain.identity) {
            throw std::runtime_error("image/no-op renderer contract is incorrect");
        }

        std::cout << "renderer=" << plain.format.renderer_id
                  << " identity=" << plain.identity << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
