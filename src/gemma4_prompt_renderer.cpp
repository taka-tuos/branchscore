#include "branchscore/gemma4_prompt_renderer.hpp"

#include "branchscore/json.hpp"

#include <array>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace branchscore {
namespace {

// A small self-contained SHA-256 implementation keeps the prompt identity
// stable without adding a runtime dependency to the independent project.
class Sha256 {
public:
    void update(std::string_view input) {
        for (const auto byte : input) {
            buffer_[buffer_size_++] = static_cast<std::uint8_t>(byte);
            if (buffer_size_ == buffer_.size()) transform();
        }
        bit_count_ += input.size() * 8U;
    }

    std::string finish() {
        buffer_[buffer_size_++] = 0x80U;
        if (buffer_size_ > 56U) {
            while (buffer_size_ < buffer_.size()) buffer_[buffer_size_++] = 0;
            transform();
        }
        while (buffer_size_ < 56U) buffer_[buffer_size_++] = 0;
        for (int shift = 56; shift >= 0; shift -= 8) {
            buffer_[buffer_size_++] = static_cast<std::uint8_t>(bit_count_ >> shift);
        }
        transform();

        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (const auto word : state_) output << std::setw(8) << word;
        return output.str();
    }

private:
    static constexpr std::array<std::uint32_t, 64> constants_ = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    };

    static std::uint32_t rotate_right(std::uint32_t value, int count) {
        return (value >> count) | (value << (32 - count));
    }

    void transform() {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t i = 0; i < 16; ++i) {
            words[i] = (static_cast<std::uint32_t>(buffer_[4 * i]) << 24U) |
                (static_cast<std::uint32_t>(buffer_[4 * i + 1]) << 16U) |
                (static_cast<std::uint32_t>(buffer_[4 * i + 2]) << 8U) |
                static_cast<std::uint32_t>(buffer_[4 * i + 3]);
        }
        for (std::size_t i = 16; i < words.size(); ++i) {
            const auto s0 = rotate_right(words[i - 15], 7) ^
                rotate_right(words[i - 15], 18) ^ (words[i - 15] >> 3U);
            const auto s1 = rotate_right(words[i - 2], 17) ^
                rotate_right(words[i - 2], 19) ^ (words[i - 2] >> 10U);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }

        auto a = state_[0];
        auto b = state_[1];
        auto c = state_[2];
        auto d = state_[3];
        auto e = state_[4];
        auto f = state_[5];
        auto g = state_[6];
        auto h = state_[7];
        for (std::size_t i = 0; i < words.size(); ++i) {
            const auto s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const auto choose = (e & f) ^ ((~e) & g);
            const auto temp1 = h + s1 + choose + constants_[i] + words[i];
            const auto s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto temp2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
        buffer_size_ = 0;
    }

    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffer_size_ = 0;
    std::uint64_t bit_count_ = 0;
    std::array<std::uint32_t, 8> state_ = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
};

constexpr std::array<std::uint32_t, 64> Sha256::constants_;

std::string prompt_identity(const std::string & text) {
    Sha256 hash;
    hash.update(text);
    return "gemma4-categorical-v1/sha256:" + hash.finish();
}

std::string answer_label(const std::size_t index) {
    if (index >= 16) throw std::runtime_error("Gemma 4 categorical readout supports A-P");
    return std::string(1, static_cast<char>('A' + index));
}

} // namespace

const char * reasoning_policy_name(ReasoningPolicy policy) noexcept {
    switch (policy) {
        case ReasoningPolicy::DirectAnswerDisabled:
            return "direct-answer-reasoning-disabled";
    }
    return "unknown";
}

const char * Gemma4PromptRenderer::renderer_id() noexcept {
    return "gemma4-categorical-v1";
}

RenderedPrompt Gemma4PromptRenderer::render(
    const std::string & state,
    const std::string & question,
    const std::vector<DecisionOption> & options,
    bool has_image,
    const PromptPolicy & policy,
    const std::optional<std::string> & requested_template_file,
    bool gguf_chat_template_present) const {
    if (state.empty()) throw std::runtime_error("state must not be empty");
    if (question.empty()) throw std::runtime_error("question must not be empty");
    if (options.size() < 2 || options.size() > 16) {
        throw std::runtime_error("categorical prompt requires 2-16 options");
    }
    if (policy.reasoning != ReasoningPolicy::DirectAnswerDisabled) {
        throw std::runtime_error("unsupported Gemma 4 reasoning policy");
    }

    std::string text;
    text.reserve(state.size() + question.size() + options.size() * 48 + 240);
    text += "<bos><|turn>system\n";
    text += "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.";
    text += "<turn|>\n<|turn>user\n";
    if (has_image) text += "<|image|>\n";
    text += "State:\n";
    text += state;
    text += "\n\nQuestion:\n";
    text += question;
    text += "\n\nOptions:\n";
    json::Value::Array rendered_options;
    rendered_options.reserve(options.size());
    std::vector<RenderedAnswerSlot> answer_slots;
    answer_slots.reserve(options.size());
    for (std::size_t index = 0; index < options.size(); ++index) {
        const auto label = answer_label(index);
        json::Value::Object rendered_option;
        rendered_option.emplace("letter", label);
        rendered_option.emplace("description", options[index].description);
        rendered_options.emplace_back(json::Value(std::move(rendered_option)));
        answer_slots.push_back(RenderedAnswerSlot{label, index, options[index].id});
    }
    text += json::stringify(json::Value(std::move(rendered_options)));
    text += "<turn|>\n<|turn>model\n";

    PromptFormatInfo format;
    format.renderer_id = renderer_id();
    format.model_family = "gemma4";
    format.effective_source = "built-in";
    format.prompt_policy = policy;
    format.requested_template_file = requested_template_file;
    format.requested_template_applied = false;
    format.gguf_chat_template_present = gguf_chat_template_present;
    format.gguf_chat_template_used = false;
    const auto identity = prompt_identity(text);
    return RenderedPrompt{
        std::move(text), std::move(format), identity, std::move(answer_slots)};
}

} // namespace branchscore
