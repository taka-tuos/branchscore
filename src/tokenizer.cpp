#include "branchscore/tokenizer.hpp"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace branchscore {
namespace {

constexpr std::int32_t token_type_unknown = 2;
constexpr std::int32_t token_type_control = 3;
constexpr std::int32_t token_type_user_defined = 4;

struct PairHash {
    std::size_t operator()(const std::pair<std::string, std::string> & value) const noexcept {
        return std::hash<std::string>{}(value.first) ^
               (std::hash<std::string>{}(value.second) << 1U);
    }
};

struct SpecialToken {
    std::string text;
    TokenId id = -1;
    std::int32_t type = 0;
};

struct Symbol {
    int previous = -1;
    int next = -1;
    std::size_t offset = 0;
    std::size_t size = 0;
};

struct Bigram {
    int left = -1;
    int right = -1;
    int rank = -1;
    std::string text;
};

struct BigramLater {
    bool operator()(const Bigram & left, const Bigram & right) const noexcept {
        return left.rank > right.rank ||
               (left.rank == right.rank && left.left > right.left);
    }
};

class GgufMetadata {
public:
    explicit GgufMetadata(const std::string & path) {
        gguf_init_params params{/* no_alloc = */ true, /* ctx = */ &tensor_context_};
        context_ = gguf_init_from_file(path.c_str(), params);
        if (context_ == nullptr) {
            throw std::runtime_error("failed to read tokenizer metadata from '" + path + "'");
        }
    }

    ~GgufMetadata() {
        if (tensor_context_ != nullptr) ggml_free(tensor_context_);
        if (context_ != nullptr) gguf_free(context_);
    }

    const gguf_context * get() const noexcept { return context_; }

private:
    gguf_context * context_ = nullptr;
    ggml_context * tensor_context_ = nullptr;
};

int64_t require_key(const gguf_context * context, const std::string & key) {
    const auto id = gguf_find_key(context, key.c_str());
    if (id < 0) throw std::runtime_error("required tokenizer key is missing: " + key);
    return id;
}

std::string require_string(const gguf_context * context, const std::string & key) {
    const auto id = require_key(context, key);
    if (gguf_get_kv_type(context, id) != GGUF_TYPE_STRING) {
        throw std::runtime_error("tokenizer key has wrong type: " + key);
    }
    return gguf_get_val_str(context, id);
}

TokenId optional_token_id(
    const gguf_context * context,
    const std::string & key,
    TokenId fallback = -1) {
    const auto id = gguf_find_key(context, key.c_str());
    if (id < 0) return fallback;
    std::uint64_t value = 0;
    switch (gguf_get_kv_type(context, id)) {
        case GGUF_TYPE_UINT32: value = gguf_get_val_u32(context, id); break;
        case GGUF_TYPE_INT32: {
            const auto signed_value = gguf_get_val_i32(context, id);
            if (signed_value < 0) throw std::runtime_error("negative tokenizer ID: " + key);
            value = static_cast<std::uint64_t>(signed_value);
            break;
        }
        case GGUF_TYPE_UINT64: value = gguf_get_val_u64(context, id); break;
        default: throw std::runtime_error("tokenizer ID has wrong type: " + key);
    }
    if (value > static_cast<std::uint64_t>(std::numeric_limits<TokenId>::max())) {
        throw std::runtime_error("tokenizer ID is out of range: " + key);
    }
    return static_cast<TokenId>(value);
}

std::size_t utf8_character_size(unsigned char first, std::size_t remaining) {
    std::size_t expected = 1;
    if ((first & 0xE0U) == 0xC0U) expected = 2;
    else if ((first & 0xF0U) == 0xE0U) expected = 3;
    else if ((first & 0xF8U) == 0xF0U) expected = 4;
    return std::min(expected, remaining);
}

std::string escape_spaces(const std::string & text) {
    static constexpr std::array<char, 3> marker{
        static_cast<char>(0xE2), static_cast<char>(0x96), static_cast<char>(0x81)};
    std::string result;
    result.reserve(text.size());
    for (char byte : text) {
        if (byte == ' ') result.append(marker.data(), marker.size());
        else result.push_back(byte);
    }
    return result;
}

std::string byte_piece(unsigned char byte) {
    static constexpr char hex[] = "0123456789ABCDEF";
    std::string result = "<0x00>";
    result[3] = hex[byte >> 4U];
    result[4] = hex[byte & 0x0FU];
    return result;
}

} // namespace

struct GemmaTokenizer::Impl {
    std::vector<std::string> id_to_piece;
    std::vector<std::int32_t> token_types;
    std::unordered_map<std::string, TokenId> piece_to_id;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merge_ranks;
    std::vector<SpecialToken> special_tokens;
    TokenId bos = -1;
    TokenId eos = -1;
    std::optional<std::string> chat_template;

    std::vector<TokenId> tokenize_raw(const std::string & raw) const {
        const std::string text = escape_spaces(raw);
        std::vector<TokenId> output;
        std::size_t segment_start = 0;
        while (segment_start < text.size()) {
            const bool newline = text[segment_start] == '\n';
            std::size_t segment_end = segment_start + 1;
            while (segment_end < text.size() && (text[segment_end] == '\n') == newline) {
                ++segment_end;
            }
            tokenize_segment(text.substr(segment_start, segment_end - segment_start), output);
            segment_start = segment_end;
        }
        return output;
    }

    void tokenize_segment(const std::string & segment, std::vector<TokenId> & output) const {
        if (segment.empty()) return;
        if (segment.find_first_not_of('\n') == std::string::npos) {
            const auto whole = piece_to_id.find(segment);
            if (whole != piece_to_id.end()) {
                output.push_back(whole->second);
                return;
            }
        }

        std::vector<Symbol> symbols;
        for (std::size_t offset = 0; offset < segment.size();) {
            const auto size = utf8_character_size(
                static_cast<unsigned char>(segment[offset]), segment.size() - offset);
            const int index = static_cast<int>(symbols.size());
            symbols.push_back(Symbol{
                index - 1,
                offset + size == segment.size() ? -1 : index + 1,
                offset,
                size,
            });
            offset += size;
        }

        std::priority_queue<Bigram, std::vector<Bigram>, BigramLater> queue;
        const auto add_bigram = [&](int left, int right) {
            if (left < 0 || right < 0) return;
            const auto left_text = segment.substr(symbols[left].offset, symbols[left].size);
            const auto right_text = segment.substr(symbols[right].offset, symbols[right].size);
            const auto rank = merge_ranks.find({left_text, right_text});
            if (rank == merge_ranks.end()) return;
            queue.push(Bigram{left, right, rank->second, left_text + right_text});
        };
        for (int i = 1; i < static_cast<int>(symbols.size()); ++i) {
            add_bigram(i - 1, i);
        }

        while (!queue.empty()) {
            Bigram bigram = queue.top();
            queue.pop();
            auto & left = symbols[bigram.left];
            auto & right = symbols[bigram.right];
            if (left.size == 0 || right.size == 0 ||
                segment.substr(left.offset, left.size + right.size) != bigram.text) {
                continue;
            }
            left.size += right.size;
            right.size = 0;
            left.next = right.next;
            if (right.next >= 0) symbols[right.next].previous = bigram.left;
            add_bigram(left.previous, bigram.left);
            add_bigram(bigram.left, left.next);
        }

        for (int index = 0; index >= 0; index = symbols[index].next) {
            const auto & symbol = symbols[index];
            const auto text = segment.substr(symbol.offset, symbol.size);
            const auto token = piece_to_id.find(text);
            if (token != piece_to_id.end()) {
                output.push_back(token->second);
                continue;
            }
            for (unsigned char byte : text) {
                const auto fallback = piece_to_id.find(byte_piece(byte));
                if (fallback == piece_to_id.end()) {
                    throw std::runtime_error("tokenizer has no byte fallback for input");
                }
                output.push_back(fallback->second);
            }
        }
    }
};

GemmaTokenizer GemmaTokenizer::from_gguf(const std::string & model_path) {
    GgufMetadata metadata(model_path);
    const auto * context = metadata.get();
    if (require_string(context, "tokenizer.ggml.model") != "gemma4") {
        throw std::runtime_error("GGUF does not contain a Gemma 4 tokenizer");
    }

    auto impl = std::make_unique<Impl>();
    const auto tokens_id = require_key(context, "tokenizer.ggml.tokens");
    if (gguf_get_kv_type(context, tokens_id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(context, tokens_id) != GGUF_TYPE_STRING) {
        throw std::runtime_error("tokenizer.ggml.tokens is not a string array");
    }
    const auto token_count = gguf_get_arr_n(context, tokens_id);
    impl->id_to_piece.reserve(token_count);
    impl->piece_to_id.reserve(token_count);
    for (std::size_t i = 0; i < token_count; ++i) {
        std::string piece = gguf_get_arr_str(context, tokens_id, i);
        impl->id_to_piece.push_back(piece);
        impl->piece_to_id.emplace(std::move(piece), static_cast<TokenId>(i));
    }
    if (impl->piece_to_id.size() != impl->id_to_piece.size()) {
        throw std::runtime_error("tokenizer contains duplicate pieces");
    }

    const auto types_id = require_key(context, "tokenizer.ggml.token_type");
    if (gguf_get_kv_type(context, types_id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(context, types_id) != GGUF_TYPE_INT32 ||
        gguf_get_arr_n(context, types_id) < token_count) {
        throw std::runtime_error("tokenizer.ggml.token_type is invalid");
    }
    const auto * types = static_cast<const std::int32_t *>(gguf_get_arr_data(context, types_id));
    impl->token_types.assign(types, types + token_count);

    const auto merges_id = require_key(context, "tokenizer.ggml.merges");
    if (gguf_get_kv_type(context, merges_id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(context, merges_id) != GGUF_TYPE_STRING) {
        throw std::runtime_error("tokenizer.ggml.merges is not a string array");
    }
    const auto merge_count = gguf_get_arr_n(context, merges_id);
    impl->merge_ranks.reserve(merge_count);
    for (std::size_t i = 0; i < merge_count; ++i) {
        const std::string merge = gguf_get_arr_str(context, merges_id, i);
        const auto separator = merge.find(' ', 1);
        if (separator == std::string::npos) {
            throw std::runtime_error("invalid Gemma 4 BPE merge entry");
        }
        impl->merge_ranks.emplace(
            std::make_pair(merge.substr(0, separator), merge.substr(separator + 1)),
            static_cast<int>(i));
    }

    impl->bos = optional_token_id(context, "tokenizer.ggml.bos_token_id");
    impl->eos = optional_token_id(context, "tokenizer.ggml.eos_token_id");
    if (impl->bos < 0 || impl->eos < 0 ||
        static_cast<std::size_t>(impl->bos) >= token_count ||
        static_cast<std::size_t>(impl->eos) >= token_count) {
        throw std::runtime_error("Gemma 4 BOS/EOS IDs are missing or invalid");
    }
    const auto chat_template_id = gguf_find_key(context, "tokenizer.chat_template");
    if (chat_template_id >= 0) {
        if (gguf_get_kv_type(context, chat_template_id) != GGUF_TYPE_STRING) {
            throw std::runtime_error("tokenizer.chat_template has wrong type");
        }
        impl->chat_template = gguf_get_val_str(context, chat_template_id);
    }

    for (std::size_t i = 0; i < token_count; ++i) {
        auto type = impl->token_types[i];
        const auto & piece = impl->id_to_piece[i];
        if (piece == "<eos>" || piece == "<turn|>" || piece == "<|tool_response>") {
            type = token_type_control;
            impl->token_types[i] = type;
        }
        if (type == token_type_control || type == token_type_user_defined ||
            type == token_type_unknown) {
            impl->special_tokens.push_back(
                SpecialToken{impl->id_to_piece[i], static_cast<TokenId>(i), type});
        }
    }
    std::stable_sort(
        impl->special_tokens.begin(), impl->special_tokens.end(),
        [](const auto & left, const auto & right) { return left.text.size() > right.text.size(); });
    return GemmaTokenizer(std::move(impl));
}

GemmaTokenizer::GemmaTokenizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GemmaTokenizer::~GemmaTokenizer() = default;
GemmaTokenizer::GemmaTokenizer(GemmaTokenizer &&) noexcept = default;
GemmaTokenizer & GemmaTokenizer::operator=(GemmaTokenizer &&) noexcept = default;

std::vector<TokenId> GemmaTokenizer::tokenize(
    const std::string & text,
    bool add_bos,
    bool parse_special) const {
    std::vector<TokenId> output;
    if (add_bos) output.push_back(impl_->bos);

    std::size_t position = 0;
    while (position < text.size()) {
        std::size_t match_position = std::string::npos;
        const SpecialToken * match_token = nullptr;
        for (const auto & special : impl_->special_tokens) {
            if (!parse_special && special.type != token_type_user_defined) continue;
            const auto found = text.find(special.text, position);
            if (found < match_position) {
                match_position = found;
                match_token = &special;
            }
        }

        const auto raw_end = match_token == nullptr ? text.size() : match_position;
        if (raw_end > position) {
            auto raw_ids = impl_->tokenize_raw(text.substr(position, raw_end - position));
            output.insert(output.end(), raw_ids.begin(), raw_ids.end());
        }
        if (match_token == nullptr) break;
        output.push_back(match_token->id);
        position = match_position + match_token->text.size();
    }
    return output;
}

AnswerToken GemmaTokenizer::tokenize_answer_label(
    const std::string & rendered_prompt,
    const std::string & label) const {
    if (label.size() != 1 || label[0] < 'A' || label[0] > 'P') {
        throw std::runtime_error("Gemma 4 answer label must be one uppercase letter A-P");
    }
    const bool rendered_has_bos = rendered_prompt.rfind("<bos>", 0) == 0;
    const auto prefix_ids = tokenize(rendered_prompt, !rendered_has_bos, true);
    const auto standalone_ids = tokenize(label, false, true);
    if (standalone_ids.size() != 1) {
        throw std::runtime_error("answer label '" + label + "' is not a single token");
    }
    const auto id = standalone_ids.front();
    if (piece(id) != label || id == eos_id() || is_special_token(id)) {
        throw std::runtime_error("answer label '" + label + "' is not a normal token");
    }
    const auto combined_ids = tokenize(rendered_prompt + label, !rendered_has_bos, true);
    if (combined_ids.size() != prefix_ids.size() + 1 ||
        !std::equal(prefix_ids.begin(), prefix_ids.end(), combined_ids.begin()) ||
        combined_ids.back() != id) {
        throw std::runtime_error(
            "answer label '" + label + "' changes tokenization at the prompt boundary");
    }
    return AnswerToken{label, id, true};
}

const std::string & GemmaTokenizer::piece(TokenId id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= impl_->id_to_piece.size()) {
        throw std::out_of_range("token ID is outside the vocabulary");
    }
    return impl_->id_to_piece[static_cast<std::size_t>(id)];
}

std::size_t GemmaTokenizer::vocabulary_size() const noexcept {
    return impl_->id_to_piece.size();
}

TokenId GemmaTokenizer::bos_id() const noexcept { return impl_->bos; }
TokenId GemmaTokenizer::eos_id() const noexcept { return impl_->eos; }

std::optional<std::string> GemmaTokenizer::chat_template() const noexcept {
    return impl_->chat_template;
}

std::optional<TokenId> GemmaTokenizer::find_token(const std::string & piece) const {
    const auto found = impl_->piece_to_id.find(piece);
    if (found == impl_->piece_to_id.end()) return std::nullopt;
    return found->second;
}

bool GemmaTokenizer::is_special_token(const TokenId id) const {
    return std::any_of(
        impl_->special_tokens.begin(), impl_->special_tokens.end(),
        [id](const SpecialToken & token) { return token.id == id; });
}

} // namespace branchscore
