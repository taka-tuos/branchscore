#pragma once

#include <cstdint>
#include <optional>
#include <memory>
#include <string>
#include <vector>

namespace branchscore {

using TokenId = std::int32_t;

struct OptionTokens {
    std::string option_id;
    std::size_t input_index = 0;
    std::string description;
    std::vector<TokenId> ids;
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

    OptionTokens tokenize_option(
        const std::string & rendered_prefix,
        const std::string & option_id,
        std::size_t input_index,
        const std::string & description) const;

    const std::string & piece(TokenId id) const;
    std::size_t vocabulary_size() const noexcept;
    TokenId bos_id() const noexcept;
    TokenId eos_id() const noexcept;
    const std::string & chat_template() const noexcept;
    std::optional<TokenId> find_token(const std::string & piece) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    explicit GemmaTokenizer(std::unique_ptr<Impl> impl);
};

} // namespace branchscore
