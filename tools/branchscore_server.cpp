#include "branchscore/backend_context.hpp"
#include "branchscore/gemma4_decision_engine.hpp"
#include "branchscore/image_preprocessor.hpp"
#include "branchscore/json.hpp"
#include "branchscore/model_loader.hpp"
#include "branchscore/systemone_adapter.hpp"
#include "branchscore/tokenizer.hpp"

#include <llhttp.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;
using Json = branchscore::json::Value;
using Object = Json::Object;

constexpr std::size_t max_request_body_bytes = 16U * 1024U * 1024U;
constexpr std::size_t max_request_header_bytes = 16U * 1024U;
constexpr std::size_t max_request_header_count = 64;
constexpr std::uint32_t max_image_dimension = 8192;
constexpr std::uint64_t max_decoded_image_pixels = 8U * 1024U * 1024U;
constexpr auto connection_timeout = std::chrono::seconds(10);
constexpr auto send_timeout = std::chrono::seconds(10);
constexpr const char * bearer_token_environment = "BRANCHSCORE_BEARER_TOKEN";

volatile std::sig_atomic_t stopping = 0;

void on_signal(int) { stopping = 1; }

class FileDescriptor {
public:
    explicit FileDescriptor(const int value = -1) : value_(value) {}
    ~FileDescriptor() { reset(); }
    FileDescriptor(const FileDescriptor &) = delete;
    FileDescriptor & operator=(const FileDescriptor &) = delete;
    FileDescriptor(FileDescriptor && other) noexcept : value_(other.release()) {}
    FileDescriptor & operator=(FileDescriptor && other) noexcept {
        if (this != &other) reset(other.release());
        return *this;
    }
    int get() const noexcept { return value_; }
    explicit operator bool() const noexcept { return value_ >= 0; }
    int release() noexcept {
        const auto value = value_;
        value_ = -1;
        return value;
    }
    void reset(const int value = -1) noexcept {
        if (value_ >= 0) ::close(value_);
        value_ = value;
    }

private:
    int value_;
};

struct Config {
    std::string model_path;
    std::string mmproj_path;
    std::string backend = "auto";
    std::string host = "127.0.0.1";
    std::uint16_t port = 8080;
};

struct HttpRequest {
    std::string method;
    std::string url;
    std::string content_type;
    std::string authorization;
    std::string content_encoding;
    std::string expect;
    std::string body;
};

struct ParseContext {
    HttpRequest request;
    std::string header_name;
    std::string header_value;
    std::size_t header_bytes = 0;
    std::size_t header_count = 0;
    std::size_t authorization_count = 0;
    std::size_t content_type_count = 0;
    int failure_status = 0;
    std::string failure_code;
    bool complete = false;
    bool require_bearer = false;
    const std::string * bearer_token = nullptr;
};

struct ParseResult {
    HttpRequest request;
    int status = 0;
    std::string error_code;
};

std::string lowercase(std::string value) {
    for (auto & c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

std::string trim(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](const unsigned char c) {
        return std::isspace(c) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](const unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    if (first >= last) return {};
    return std::string(first, last);
}

bool add_header_bytes(ParseContext & context, const std::size_t count) {
    if (count > max_request_header_bytes - context.header_bytes) {
        context.failure_status = 431;
        context.failure_code = "headers_too_large";
        return false;
    }
    context.header_bytes += count;
    return true;
}

int append_request_line_part(
    llhttp_t * parser,
    const char * bytes,
    const std::size_t length,
    std::string & destination,
    const std::size_t max_length) {
    auto & context = *static_cast<ParseContext *>(parser->data);
    if (length > max_length - std::min(max_length, destination.size())) {
        context.failure_status = 414;
        context.failure_code = "request_target_too_large";
        return HPE_USER;
    }
    if (!add_header_bytes(context, length)) return HPE_USER;
    destination.append(bytes, length);
    return HPE_OK;
}

int on_method(llhttp_t * parser, const char * bytes, const std::size_t length) {
    auto & context = *static_cast<ParseContext *>(parser->data);
    return append_request_line_part(parser, bytes, length, context.request.method, 32);
}

int on_url(llhttp_t * parser, const char * bytes, const std::size_t length) {
    auto & context = *static_cast<ParseContext *>(parser->data);
    return append_request_line_part(parser, bytes, length, context.request.url, 4096);
}

int on_version(llhttp_t * parser, const char * bytes, const std::size_t length) {
    auto & context = *static_cast<ParseContext *>(parser->data);
    if (!add_header_bytes(context, length)) return HPE_USER;
    return HPE_OK;
}

int on_header_field(llhttp_t * parser, const char * bytes, const std::size_t length) {
    auto & context = *static_cast<ParseContext *>(parser->data);
    if (length > max_request_header_bytes -
                     std::min(max_request_header_bytes, context.header_name.size())) {
        context.failure_status = 431;
        context.failure_code = "headers_too_large";
        return HPE_USER;
    }
    if (!add_header_bytes(context, length)) return HPE_USER;
    context.header_name.append(bytes, length);
    return HPE_OK;
}

int on_header_field_complete(llhttp_t * parser) {
    auto & context = *static_cast<ParseContext *>(parser->data);
    context.header_name = lowercase(std::move(context.header_name));
    return HPE_OK;
}

int on_header_value(llhttp_t * parser, const char * bytes, const std::size_t length) {
    auto & context = *static_cast<ParseContext *>(parser->data);
    if (length > max_request_header_bytes -
                     std::min(max_request_header_bytes, context.header_value.size())) {
        context.failure_status = 431;
        context.failure_code = "headers_too_large";
        return HPE_USER;
    }
    if (!add_header_bytes(context, length)) return HPE_USER;
    context.header_value.append(bytes, length);
    return HPE_OK;
}

int on_header_value_complete(llhttp_t * parser) {
    auto & context = *static_cast<ParseContext *>(parser->data);
    ++context.header_count;
    if (!add_header_bytes(context, 4) || context.header_count > max_request_header_count) {
        context.failure_status = 431;
        context.failure_code = "headers_too_large";
        return HPE_USER;
    }
    const auto value = trim(std::move(context.header_value));
    if (context.header_name == "content-type") {
        ++context.content_type_count;
        context.request.content_type = value;
    } else if (context.header_name == "authorization") {
        ++context.authorization_count;
        context.request.authorization = value;
    } else if (context.header_name == "content-encoding") {
        context.request.content_encoding = lowercase(value);
    } else if (context.header_name == "expect") {
        context.request.expect = lowercase(value);
    }
    context.header_name.clear();
    context.header_value.clear();
    return HPE_OK;
}

bool constant_time_equal(const std::string & left, const std::string & right) {
    const auto count = std::max(left.size(), right.size());
    std::uint32_t difference = static_cast<std::uint32_t>(left.size() ^ right.size());
    for (std::size_t index = 0; index < count; ++index) {
        const auto a = index < left.size()
            ? static_cast<unsigned char>(left[index]) : 0U;
        const auto b = index < right.size()
            ? static_cast<unsigned char>(right[index]) : 0U;
        difference |= static_cast<std::uint32_t>(a ^ b);
    }
    return difference == 0;
}

bool authorized(const std::string & value, const std::string & expected_token) {
    const auto space = value.find(' ');
    if (space == std::string::npos || lowercase(value.substr(0, space)) != "bearer") {
        return false;
    }
    const auto supplied = trim(value.substr(space + 1));
    return !supplied.empty() && constant_time_equal(supplied, expected_token);
}

int on_headers_complete(llhttp_t * parser) {
    auto & context = *static_cast<ParseContext *>(parser->data);
    if (context.authorization_count > 1 || context.content_type_count > 1) {
        context.failure_status = 400;
        context.failure_code = "invalid_http_request";
        return -1;
    }
    if (context.require_bearer &&
        (context.bearer_token == nullptr ||
         !authorized(context.request.authorization, *context.bearer_token))) {
        context.failure_status = 401;
        context.failure_code = "unauthorized";
        return -1;
    }
    if (parser->content_length > max_request_body_bytes) {
        context.failure_status = 413;
        context.failure_code = "request_too_large";
        return -1;
    }
    if (!context.request.expect.empty()) {
        context.failure_status = 417;
        context.failure_code = "expectation_not_supported";
        return -1;
    }
    if (!context.request.content_encoding.empty() &&
        context.request.content_encoding != "identity") {
        context.failure_status = 415;
        context.failure_code = "unsupported_content_encoding";
        return -1;
    }
    return 0;
}

int on_body(llhttp_t * parser, const char * bytes, const std::size_t length) {
    auto & context = *static_cast<ParseContext *>(parser->data);
    if (length > max_request_body_bytes - context.request.body.size()) {
        context.failure_status = 413;
        context.failure_code = "request_too_large";
        return HPE_USER;
    }
    context.request.body.append(bytes, length);
    return HPE_OK;
}

int on_message_complete(llhttp_t * parser) {
    auto & context = *static_cast<ParseContext *>(parser->data);
    context.complete = true;
    return HPE_PAUSED;
}

bool wait_for(const int socket, const short events, const Clock::time_point deadline) {
    while (true) {
        const auto remaining = deadline - Clock::now();
        if (remaining <= Clock::duration::zero()) return false;
        const auto timeout = std::chrono::duration_cast<std::chrono::milliseconds>(remaining);
        const auto timeout_ms = static_cast<int>(std::min<std::int64_t>(
            timeout.count() + 1, std::numeric_limits<int>::max()));
        pollfd descriptor{socket, events, 0};
        const int result = ::poll(&descriptor, 1, timeout_ms);
        if (result > 0) return (descriptor.revents & (events | POLLERR | POLLHUP)) != 0;
        if (result == 0) return false;
        if (errno != EINTR) return false;
    }
}

ParseResult receive_request(
    const int socket,
    const bool require_bearer,
    const std::string & bearer_token) {
    ParseContext context;
    context.require_bearer = require_bearer;
    context.bearer_token = &bearer_token;
    llhttp_settings_t settings{};
    llhttp_settings_init(&settings);
    settings.on_method = on_method;
    settings.on_url = on_url;
    settings.on_version = on_version;
    settings.on_header_field = on_header_field;
    settings.on_header_field_complete = on_header_field_complete;
    settings.on_header_value = on_header_value;
    settings.on_header_value_complete = on_header_value_complete;
    settings.on_headers_complete = on_headers_complete;
    settings.on_body = on_body;
    settings.on_message_complete = on_message_complete;
    llhttp_t parser{};
    llhttp_init(&parser, HTTP_REQUEST, &settings);
    parser.data = &context;

    std::array<char, 8192> buffer{};
    const auto deadline = Clock::now() + connection_timeout;
    while (!context.complete) {
        if (!wait_for(socket, POLLIN, deadline)) {
            if (context.failure_status == 0) {
                context.failure_status = 408;
                context.failure_code = "request_timeout";
            }
            break;
        }
        const auto count = ::recv(socket, buffer.data(), buffer.size(), 0);
        if (count == 0) {
            if (!context.complete && context.failure_status == 0) {
                const auto finish_status = llhttp_finish(&parser);
                if (finish_status != HPE_OK && !context.complete) {
                    context.failure_status = 400;
                    context.failure_code = "invalid_http_request";
                }
            }
            break;
        }
        if (count < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            context.failure_status = 400;
            context.failure_code = "invalid_http_request";
            break;
        }
        const auto parser_status = llhttp_execute(
            &parser, buffer.data(), static_cast<std::size_t>(count));
        if (context.complete) break;
        if (parser_status != HPE_OK) {
            if (context.failure_status == 0) {
                context.failure_status = 400;
                context.failure_code = "invalid_http_request";
            }
            break;
        }
    }
    if (context.complete) return {std::move(context.request), 0, {}};
    if (context.failure_status == 0) {
        context.failure_status = 400;
        context.failure_code = "invalid_http_request";
    }
    return {{}, context.failure_status, std::move(context.failure_code)};
}

std::string reason_phrase(const int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 413: return "Payload Too Large";
        case 414: return "URI Too Long";
        case 415: return "Unsupported Media Type";
        case 417: return "Expectation Failed";
        case 422: return "Unprocessable Entity";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        default: return "Error";
    }
}

bool send_all(const int socket, const std::string & bytes) {
    const auto deadline = Clock::now() + send_timeout;
    std::size_t sent = 0;
    while (sent < bytes.size()) {
        if (!wait_for(socket, POLLOUT, deadline)) return false;
        const auto count = ::send(
            socket, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
        if (count < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            return false;
        }
        if (count == 0) return false;
        sent += static_cast<std::size_t>(count);
    }
    return true;
}

bool send_json(const int socket, const int status, const Json & body) {
    const auto encoded = branchscore::json::stringify(body);
    std::string response = "HTTP/1.1 " + std::to_string(status) + " " +
        reason_phrase(status) + "\r\nContent-Type: application/json; charset=utf-8\r\n" +
        "Cache-Control: no-store\r\nConnection: close\r\n";
    if (status == 401) response += "WWW-Authenticate: Bearer realm=\"branchscore\"\r\n";
    response += "Content-Length: " + std::to_string(encoded.size()) + "\r\n\r\n" + encoded;
    return send_all(socket, response);
}

std::string public_message(const std::string & code) {
    if (code == "unauthorized") return "a valid bearer token is required";
    if (code == "unsupported_model") return "requested model is not served here";
    if (code == "unsupported_question_type") return "only Choice questions are supported";
    if (code == "unsupported_media_type") return "only JSON, PNG, and JPEG are supported";
    if (code == "invalid_image") return "image data is invalid";
    if (code == "image_too_large") return "image dimensions exceed the server limit";
    if (code == "request_too_large") return "request body exceeds the server limit";
    if (code == "request_timeout") return "request did not complete before the timeout";
    if (code == "headers_too_large") return "request headers exceed the server limit";
    if (code == "request_target_too_large") return "request target exceeds the server limit";
    if (code == "expectation_not_supported") return "Expect headers are not supported";
    if (code == "unsupported_content_encoding") return "compressed request bodies are not supported";
    if (code == "invalid_json") return "request body is not valid JSON";
    if (code == "invalid_http_request") return "HTTP request is malformed";
    if (code == "unsupported_content_type") return "Content-Type must be application/json";
    if (code == "not_found") return "route not found";
    if (code == "method_not_allowed") return "method is not supported for this route";
    return "request fields are invalid or unsupported";
}

int status_for_code(const std::string & code) {
    if (code == "unauthorized") return 401;
    if (code == "unsupported_media_type" || code == "unsupported_content_type" ||
        code == "unsupported_content_encoding") return 415;
    if (code == "request_too_large" || code == "image_too_large") return 413;
    if (code == "request_timeout") return 408;
    if (code == "headers_too_large") return 431;
    if (code == "request_target_too_large") return 414;
    if (code == "expectation_not_supported") return 417;
    if (code == "not_found") return 404;
    if (code == "method_not_allowed") return 405;
    if (code == "invalid_http_request" || code == "invalid_json") return 400;
    return 422;
}

Json error_json(const std::string & code, const std::optional<std::string> & request_id) {
    Object error;
    error.emplace("code", code);
    error.emplace("message", public_message(code));
    Object response;
    response.emplace("error", Json(std::move(error)));
    if (request_id) response.emplace("request_id", *request_id);
    return Json(std::move(response));
}

bool content_type_is_json(const std::string & content_type) {
    auto value = lowercase(trim(content_type));
    const auto delimiter = value.find(';');
    if (delimiter != std::string::npos) value = trim(value.substr(0, delimiter));
    return value == "application/json";
}

std::string safe_log_id(const std::optional<std::string> & request_id) {
    if (!request_id || request_id->empty()) return "-";
    std::string result;
    result.reserve(std::min<std::size_t>(request_id->size(), 96));
    for (const auto c : *request_id) {
        if (result.size() == 96) break;
        const auto byte = static_cast<unsigned char>(c);
        result.push_back(byte >= 0x20U && byte <= 0x7eU ? c : '?');
    }
    return result.empty() ? "-" : result;
}

void handle_connection(
    const int socket,
    const bool require_bearer,
    const std::string & bearer_token,
    branchscore::Gemma4DecisionEngine & engine,
    const std::string & model_id) {
    const auto started = Clock::now();
    std::optional<std::string> request_id;
    int status = 500;
    try {
        auto parsed = receive_request(socket, require_bearer, bearer_token);
        if (parsed.status != 0) {
            status = parsed.status;
            send_json(socket, status, error_json(parsed.error_code, request_id));
        } else if (parsed.request.url != "/v1/systemone" &&
                   parsed.request.url != "/healthz") {
            status = 404;
            send_json(socket, status, error_json("not_found", request_id));
        } else if (parsed.request.url == "/healthz" && parsed.request.method != "GET") {
            status = 405;
            send_json(socket, status, error_json("method_not_allowed", request_id));
        } else if (parsed.request.url == "/v1/systemone" &&
                   parsed.request.method != "POST") {
            status = 405;
            send_json(socket, status, error_json("method_not_allowed", request_id));
        } else if (parsed.request.url == "/healthz") {
            Object body;
            body.emplace("status", "ready");
            body.emplace("model", model_id);
            status = 200;
            send_json(socket, status, Json(std::move(body)));
        } else if (!content_type_is_json(parsed.request.content_type)) {
            status = 415;
            send_json(socket, status, error_json("unsupported_content_type", request_id));
        } else {
            Json envelope;
            try {
                envelope = branchscore::json::parse(parsed.request.body);
            } catch (const std::exception &) {
                status = 400;
                send_json(socket, status, error_json("invalid_json", request_id));
                goto request_complete;
            }
            if (const auto * supplied_id = envelope.find("request_id");
                supplied_id != nullptr && supplied_id->type() == Json::Type::string) {
                request_id = supplied_id->string();
            }

            try {
                auto request = branchscore::systemone::parse_request(envelope, model_id);
                if (request.image_bytes) {
                    const auto info = branchscore::ImagePreprocessor::inspect_encoded(
                        request.image_bytes->data(), request.image_bytes->size());
                    const auto pixels = static_cast<std::uint64_t>(info.width) * info.height;
                    if (info.width > max_image_dimension ||
                        info.height > max_image_dimension ||
                        pixels > max_decoded_image_pixels) {
                        status = 413;
                        send_json(socket, status, error_json("image_too_large", request_id));
                        goto request_complete;
                    }
                }

                std::vector<branchscore::DecisionResult> results;
                results.reserve(request.questions.size());
                for (const auto & question : request.questions) {
                    results.push_back(engine.evaluate(question.decision));
                }
                auto response = branchscore::systemone::make_response(request, results, true);
                const auto handling_ms = std::chrono::duration<double, std::milli>(
                    Clock::now() - started).count();
                auto & branchscore_metadata = response.object().at("branchscore");
                branchscore_metadata.set("http_handling_ms", Json(handling_ms));
                branchscore_metadata.set("input_tokens_include_visual", Json(false));
                status = 200;
                send_json(socket, status, response);
            } catch (const branchscore::systemone::RequestError & error) {
                status = status_for_code(error.code());
                send_json(socket, status, error_json(error.code(), request_id));
            } catch (const branchscore::ImageDecodeError &) {
                status = 422;
                send_json(socket, status, error_json("invalid_image", request_id));
            } catch (const std::exception &) {
                status = 500;
                send_json(socket, status, error_json("internal_error", request_id));
            }
        }
    } catch (const std::exception &) {
        status = 500;
        send_json(socket, status, error_json("internal_error", request_id));
    }

request_complete:
    const auto elapsed_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - started).count();
    std::cerr << "request_id=" << safe_log_id(request_id)
              << " status=" << status
              << " wall_ms=" << elapsed_ms << '\n';
}

bool is_loopback_host(const std::string & host) {
    const auto normalized = lowercase(host);
    if (normalized == "localhost") return true;
    in_addr ipv4{};
    if (::inet_pton(AF_INET, host.c_str(), &ipv4) == 1) {
        return (ntohl(ipv4.s_addr) >> 24U) == 127U;
    }
    in6_addr ipv6{};
    return ::inet_pton(AF_INET6, host.c_str(), &ipv6) == 1 &&
           IN6_IS_ADDR_LOOPBACK(&ipv6);
}

FileDescriptor create_listener(const Config & config) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    addrinfo * raw_addresses = nullptr;
    const auto port = std::to_string(config.port);
    const int lookup = ::getaddrinfo(
        config.host.c_str(), port.c_str(), &hints, &raw_addresses);
    if (lookup != 0) {
        throw std::runtime_error("failed to resolve bind host: " +
                                 std::string(gai_strerror(lookup)));
    }
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> addresses(raw_addresses, freeaddrinfo);
    for (auto * address = addresses.get(); address != nullptr; address = address->ai_next) {
        FileDescriptor listener(::socket(
            address->ai_family, address->ai_socktype, address->ai_protocol));
        if (!listener) continue;
        int reuse = 1;
        ::setsockopt(listener.get(), SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        if (address->ai_family == AF_INET6) {
            int only_v6 = 1;
            ::setsockopt(listener.get(), IPPROTO_IPV6, IPV6_V6ONLY,
                         &only_v6, sizeof(only_v6));
        }
        if (::bind(listener.get(), address->ai_addr, address->ai_addrlen) == 0 &&
            ::listen(listener.get(), 8) == 0) {
            return listener;
        }
    }
    throw std::runtime_error("failed to bind the configured HTTP address");
}

std::uint16_t parse_port(const std::string & text) {
    unsigned int value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size() || value == 0 || value > 65535) {
        throw std::runtime_error("--port must be an integer from 1 to 65535");
    }
    return static_cast<std::uint16_t>(value);
}

Config parse_arguments(const int argc, char ** argv) {
    Config config;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto next = [&](const char * option) -> std::string {
            if (index + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + option);
            }
            return argv[++index];
        };
        if (argument == "--model") config.model_path = next("--model");
        else if (argument == "--mmproj") config.mmproj_path = next("--mmproj");
        else if (argument == "--backend") config.backend = next("--backend");
        else if (argument == "--host") config.host = next("--host");
        else if (argument == "--port") config.port = parse_port(next("--port"));
        else if (argument == "--help") {
            std::cout
                << "Usage: branchscore-server --model FILE --mmproj FILE\n"
                << "       [--backend NAME] [--host ADDRESS] [--port PORT]\n"
                << "Defaults: --backend auto --host 127.0.0.1 --port 8080\n"
                << "Non-loopback binds require BRANCHSCORE_BEARER_TOKEN.\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (config.model_path.empty() || config.mmproj_path.empty()) {
        throw std::runtime_error("--model and --mmproj are required");
    }
    return config;
}

void run_server(
    const Config & config,
    const std::string & bearer_token,
    const bool require_bearer,
    branchscore::Gemma4DecisionEngine & engine) {
    auto listener = create_listener(config);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::cerr << "listening host=" << config.host
              << " port=" << config.port
              << " model=" << branchscore::systemone::default_model_id << '\n';
    while (!stopping) {
        pollfd descriptor{listener.get(), POLLIN, 0};
        const int ready = ::poll(&descriptor, 1, 500);
        if (ready < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("listener poll failed");
        }
        if (ready == 0) continue;
        sockaddr_storage address{};
        socklen_t address_size = sizeof(address);
        FileDescriptor client(::accept(
            listener.get(), reinterpret_cast<sockaddr *>(&address), &address_size));
        if (!client) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            if (stopping) break;
            throw std::runtime_error("failed to accept HTTP connection");
        }
        handle_connection(
            client.get(), require_bearer, bearer_token, engine,
            branchscore::systemone::default_model_id);
    }
}

} // namespace

int main(const int argc, char ** argv) {
    try {
        const auto config = parse_arguments(argc, argv);
        const bool require_bearer = !is_loopback_host(config.host);
        std::string bearer_token;
        if (require_bearer) {
            const auto * configured_token = std::getenv(bearer_token_environment);
            if (configured_token == nullptr || configured_token[0] == '\0') {
                throw std::runtime_error(
                    "non-loopback bind requires BRANCHSCORE_BEARER_TOKEN before model load");
            }
            bearer_token = configured_token;
            if (std::any_of(bearer_token.begin(), bearer_token.end(), [](const unsigned char c) {
                    return c < 0x21U || c > 0x7eU;
                })) {
                throw std::runtime_error("bearer token must use visible non-space ASCII");
            }
        }

        branchscore::BackendContext backend(config.backend);
        auto model = branchscore::ModelLoader::load(
            config.model_path, config.mmproj_path, backend);
        auto tokenizer = branchscore::GemmaTokenizer::from_gguf(config.model_path);
        branchscore::Gemma4DecisionEngine engine(model, backend, std::move(tokenizer));
        run_server(config, bearer_token, require_bearer, engine);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "branchscore-server: " << error.what() << '\n';
        return 1;
    }
}
