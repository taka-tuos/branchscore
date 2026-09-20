#pragma once

#include <cstdint>
#include <optional>
#include <memory>
#include <string>
#include <vector>

namespace branchscore {

using TokenId = std::int32_t;

struct AnswerToken {
    std::string label;
    TokenId id = -1;
    bool boundary_valid = false;
};

class GemmaTokenizer {
public:
    static GemmaTokenizer from_gguf(const std::string & model_path);

    ~GemmaTokenizer();
    GemmaTokenizer(const GemmaTokenizer &) = delete;
    GemmaTokenizer & operator=(const GemmaTokenizer &) = delete;
    GemmaTokenizer(GemmaTokenizer &&) noexcept;
    GemmaTokenizer & operator=(GemmaTokenizer &&) noexcept;

    std::vector<TokenId> tokenize(
        const std::string & text,
        bool add_bos,
        bool parse_special = true) const;

    AnswerToken tokenize_answer_label(
        const std::string & rendered_prompt,
        const std::string & label) const;

    const std::string & piece(TokenId id) const;
    std::size_t vocabulary_size() const noexcept;
    TokenId bos_id() const noexcept;
    TokenId eos_id() const noexcept;
    std::optional<std::string> chat_template() const noexcept;
    std::optional<TokenId> find_token(const std::string & piece) const;
    bool is_special_token(TokenId id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    explicit GemmaTokenizer(std::unique_ptr<Impl> impl);
};

} // namespace branchscore
