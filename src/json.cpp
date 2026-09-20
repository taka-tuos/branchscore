#include "branchscore/json.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace branchscore::json {
namespace {

class Parser {
public:
    explicit Parser(const std::string & text) : text_(text) {}

    Value parse_document() {
        skip_space();
        auto value = parse_value();
        skip_space();
        if (position_ != text_.size()) fail("unexpected trailing input");
        return value;
    }

private:
    [[noreturn]] void fail(const std::string & message) const {
        throw std::runtime_error(
            "invalid JSON at byte " + std::to_string(position_) + ": " + message);
    }

    void skip_space() {
        while (position_ < text_.size()) {
            const auto byte = static_cast<unsigned char>(text_[position_]);
            if (byte != ' ' && byte != '\t' && byte != '\n' && byte != '\r') return;
            ++position_;
        }
    }

    char peek() const {
        return position_ < text_.size() ? text_[position_] : '\0';
    }

    void expect(char expected) {
        if (peek() != expected) {
            fail(std::string("expected '") + expected + "'");
        }
        ++position_;
    }

    Value parse_value() {
        switch (peek()) {
            case 'n': parse_literal("null"); return Value();
            case 't': parse_literal("true"); return Value(true);
            case 'f': parse_literal("false"); return Value(false);
            case '"': return Value(parse_string());
            case '[': return parse_array();
            case '{': return parse_object();
            case '-':
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9': return Value(parse_number());
            default: fail("expected a JSON value");
        }
    }

    void parse_literal(std::string_view literal) {
        if (text_.compare(position_, literal.size(), literal) != 0) {
            fail("invalid literal");
        }
        position_ += literal.size();
    }

    static void append_utf8(std::string & result, std::uint32_t codepoint) {
        if (codepoint <= 0x7fU) {
            result.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ffU) {
            result.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
            result.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else if (codepoint <= 0xffffU) {
            result.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
            result.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            result.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else if (codepoint <= 0x10ffffU) {
            result.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
            result.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
            result.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            result.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else {
            throw std::runtime_error("invalid JSON Unicode code point");
        }
    }

    std::uint32_t parse_hex_quad() {
        if (position_ + 4 > text_.size()) fail("truncated Unicode escape");
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const auto byte = static_cast<unsigned char>(text_[position_++]);
            std::uint32_t digit = 0;
            if (byte >= '0' && byte <= '9') digit = byte - '0';
            else if (byte >= 'a' && byte <= 'f') digit = byte - 'a' + 10U;
            else if (byte >= 'A' && byte <= 'F') digit = byte - 'A' + 10U;
            else fail("invalid Unicode escape");
            value = (value << 4U) | digit;
        }
        return value;
    }

    std::string parse_string() {
        expect('"');
        std::string result;
        while (position_ < text_.size()) {
            const auto byte = static_cast<unsigned char>(text_[position_++]);
            if (byte == '"') return result;
            if (byte < 0x20U) fail("control character in string");
            if (byte != '\\') {
                result.push_back(static_cast<char>(byte));
                continue;
            }

            if (position_ >= text_.size()) fail("truncated string escape");
            const char escape = text_[position_++];
            switch (escape) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'u': {
                    const auto first = parse_hex_quad();
                    std::uint32_t codepoint = first;
                    if (first >= 0xd800U && first <= 0xdbffU) {
                        if (position_ + 6 > text_.size() || text_[position_] != '\\' ||
                            text_[position_ + 1] != 'u') {
                            fail("unpaired high surrogate");
                        }
                        position_ += 2;
                        const auto second = parse_hex_quad();
                        if (second < 0xdc00U || second > 0xdfffU) {
                            fail("invalid low surrogate");
                        }
                        codepoint = 0x10000U + ((first - 0xd800U) << 10U) +
                                    (second - 0xdc00U);
                    } else if (first >= 0xdc00U && first <= 0xdfffU) {
                        fail("unpaired low surrogate");
                    }
                    append_utf8(result, codepoint);
                    break;
                }
                default: fail("unknown string escape");
            }
        }
        fail("unterminated string");
    }

    double parse_number() {
        const auto start = position_;
        if (peek() == '-') ++position_;
        if (peek() == '0') {
            ++position_;
            if (peek() >= '0' && peek() <= '9') fail("leading zero in number");
        } else {
            if (peek() < '1' || peek() > '9') fail("invalid number");
            while (peek() >= '0' && peek() <= '9') ++position_;
        }
        if (peek() == '.') {
            ++position_;
            if (peek() < '0' || peek() > '9') fail("missing fraction digits");
            while (peek() >= '0' && peek() <= '9') ++position_;
        }
        if (peek() == 'e' || peek() == 'E') {
            ++position_;
            if (peek() == '+' || peek() == '-') ++position_;
            if (peek() < '0' || peek() > '9') fail("missing exponent digits");
            while (peek() >= '0' && peek() <= '9') ++position_;
        }

        const auto text = text_.substr(start, position_ - start);
        char * end = nullptr;
        const auto value = std::strtod(text.c_str(), &end);
        if (end == text.c_str() || *end != '\0' || !std::isfinite(value)) {
            fail("number is not finite");
        }
        return value;
    }

    Value parse_array() {
        expect('[');
        skip_space();
        Value::Array result;
        if (peek() == ']') {
            ++position_;
            return Value(std::move(result));
        }
        while (true) {
            result.push_back(parse_value());
            skip_space();
            if (peek() == ']') {
                ++position_;
                return Value(std::move(result));
            }
            expect(',');
            skip_space();
        }
    }

    Value parse_object() {
        expect('{');
        skip_space();
        Value::Object result;
        if (peek() == '}') {
            ++position_;
            return Value(std::move(result));
        }
        while (true) {
            if (peek() != '"') fail("object key must be a string");
            const auto key = parse_string();
            skip_space();
            expect(':');
            skip_space();
            if (!result.emplace(key, parse_value()).second) {
                fail("duplicate object key");
            }
            skip_space();
            if (peek() == '}') {
                ++position_;
                return Value(std::move(result));
            }
            expect(',');
            skip_space();
        }
    }

    const std::string & text_;
    std::size_t position_ = 0;
};

void append_escaped(std::string & output, const std::string & value) {
    output.push_back('"');
    for (const auto byte : value) {
        switch (byte) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (static_cast<unsigned char>(byte) < 0x20U) {
                    std::ostringstream escaped;
                    escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                            << static_cast<unsigned int>(static_cast<unsigned char>(byte));
                    output += escaped.str();
                } else {
                    output.push_back(byte);
                }
        }
    }
    output.push_back('"');
}

void append_value(std::string & output, const Value & value) {
    switch (value.type()) {
        case Value::Type::null_value:
            output += "null";
            return;
        case Value::Type::boolean:
            output += value.boolean() ? "true" : "false";
            return;
        case Value::Type::number: {
            if (!std::isfinite(value.number())) {
                throw std::runtime_error("cannot encode a non-finite JSON number");
            }
            std::ostringstream encoded;
            encoded << std::setprecision(17) << value.number();
            output += encoded.str();
            return;
        }
        case Value::Type::string:
            append_escaped(output, value.string());
            return;
        case Value::Type::array: {
            output.push_back('[');
            bool first = true;
            for (const auto & item : value.array()) {
                if (!first) output.push_back(',');
                first = false;
                append_value(output, item);
            }
            output.push_back(']');
            return;
        }
        case Value::Type::object: {
            output.push_back('{');
            bool first = true;
            for (const auto & [key, item] : value.object()) {
                if (!first) output.push_back(',');
                first = false;
                append_escaped(output, key);
                output.push_back(':');
                append_value(output, item);
            }
            output.push_back('}');
            return;
        }
    }
    throw std::runtime_error("unknown JSON value type");
}

} // namespace

Value::Value() = default;

Value::Value(const bool value) : type_(Type::boolean), boolean_(value) {}

Value::Value(const double value) : type_(Type::number), number_(value) {}

Value::Value(std::string value) : type_(Type::string), string_(std::move(value)) {}

Value::Value(const char * value) : Value(std::string(value)) {}

Value::Value(Array value) : type_(Type::array), array_(std::move(value)) {}

Value::Value(Object value) : type_(Type::object), object_(std::move(value)) {}

Value::Type Value::type() const noexcept { return type_; }

bool Value::boolean() const {
    if (type_ != Type::boolean) throw std::runtime_error("JSON value is not a boolean");
    return boolean_;
}

double Value::number() const {
    if (type_ != Type::number) throw std::runtime_error("JSON value is not a number");
    return number_;
}

const std::string & Value::string() const {
    if (type_ != Type::string) throw std::runtime_error("JSON value is not a string");
    return string_;
}

const Value::Array & Value::array() const {
    if (type_ != Type::array) throw std::runtime_error("JSON value is not an array");
    return array_;
}

const Value::Object & Value::object() const {
    if (type_ != Type::object) throw std::runtime_error("JSON value is not an object");
    return object_;
}

Value::Object & Value::object() {
    if (type_ != Type::object) throw std::runtime_error("JSON value is not an object");
    return object_;
}

const Value * Value::find(const std::string & key) const noexcept {
    if (type_ != Type::object) return nullptr;
    const auto found = object_.find(key);
    return found == object_.end() ? nullptr : &found->second;
}

void Value::set(std::string key, Value value) {
    if (type_ != Type::object) throw std::runtime_error("JSON value is not an object");
    object_[std::move(key)] = std::move(value);
}

Value parse(const std::string & text) { return Parser(text).parse_document(); }

std::string stringify(const Value & value) {
    std::string result;
    append_value(result, value);
    return result;
}

} // namespace branchscore::json
