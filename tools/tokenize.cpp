#include "branchscore/tokenizer.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void dump(
    const branchscore::GemmaTokenizer & tokenizer,
    const std::vector<branchscore::TokenId> & ids) {
    std::cout << "token_count=" << ids.size() << '\n';
    for (std::size_t i = 0; i < ids.size(); ++i) {
        std::cout << i << '\t' << ids[i] << '\t' << tokenizer.piece(ids[i]) << '\n';
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        std::string model;
        std::string text;
        std::string prefix;
        std::string answer_label;
        bool add_bos = true;
        bool parse_special = true;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--model" && i + 1 < argc) model = argv[++i];
            else if (arg == "--text" && i + 1 < argc) text = argv[++i];
            else if (arg == "--prefix" && i + 1 < argc) prefix = argv[++i];
            else if (arg == "--answer-label" && i + 1 < argc) {
                answer_label = argv[++i];
            }
            else if (arg == "--no-bos") add_bos = false;
            else if (arg == "--no-parse-special") parse_special = false;
            else if (arg == "--help") {
                std::cout << "Usage: branchscore-tokenize --model FILE --text TEXT [--no-bos]\n"
                             "       branchscore-tokenize --model FILE --prefix TEXT --answer-label A\n";
                return 0;
            } else {
                throw std::runtime_error("unknown or incomplete argument: " + arg);
            }
        }
        if (model.empty()) throw std::runtime_error("--model is required");
        auto tokenizer = branchscore::GemmaTokenizer::from_gguf(model);
        std::cout << "vocab=" << tokenizer.vocabulary_size()
                  << " bos=" << tokenizer.bos_id()
                  << " eos=" << tokenizer.eos_id() << '\n';
        if (!prefix.empty() || !answer_label.empty()) {
            if (prefix.empty() || answer_label.empty()) {
                throw std::runtime_error("--prefix and --answer-label must be supplied together");
            }
            const auto token = tokenizer.tokenize_answer_label(prefix, answer_label);
            dump(tokenizer, {token.id});
        } else {
            dump(tokenizer, tokenizer.tokenize(text, add_bos, parse_special));
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "branchscore-tokenize: " << error.what() << '\n';
        return 1;
    }
}
