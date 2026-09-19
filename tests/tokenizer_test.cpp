#include "branchscore/tokenizer.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool expect(
    const std::vector<branchscore::TokenId> & actual,
    const std::vector<branchscore::TokenId> & expected,
    const char * label) {
    if (actual == expected) return true;
    std::cerr << label << " token IDs differ\n";
    return false;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2 || std::string(argv[1]).empty()) {
        std::cout << "BRANCHSCORE_TEST_MODEL is not configured; skipping tokenizer test\n";
        return 77;
    }

    auto tokenizer = branchscore::GemmaTokenizer::from_gguf(argv[1]);
    bool valid = true;
    valid &= tokenizer.vocabulary_size() == 262144;
    valid &= tokenizer.bos_id() == 2;
    valid &= tokenizer.eos_id() == 106;
    valid &= expect(tokenizer.tokenize("Hello world", true), {2, 9259, 1902}, "English");
    valid &= expect(
        tokenizer.tokenize("状態を見て判断する。", true),
        {2, 34593, 90166, 38303, 4042, 236924},
        "Japanese");
    valid &= expect(
        tokenizer.tokenize("alpha\n\nbeta", true),
        {2, 2482, 108, 3357},
        "newlines");
    valid &= expect(
        tokenizer.tokenize("<eos><turn|><|tool_response></s>", true),
        {2, 1, 106, 50, 954, 236751, 236813},
        "special tokens");

    const auto option = tokenizer.tokenize_option(
        "<|turn>model\n", "answer", 0, "状態を見て判断する。");
    valid &= option.boundary_valid;
    valid &= expect(option.ids, {34593, 90166, 38303, 4042, 236924}, "option");

    try {
        (void) tokenizer.tokenize_option("Hell", "merged", 1, "o");
        std::cerr << "changed prefix boundary was accepted\n";
        valid = false;
    } catch (const std::runtime_error &) {
    }
    return valid ? 0 : 1;
}

