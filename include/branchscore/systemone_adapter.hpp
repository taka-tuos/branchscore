#pragma once

#include "branchscore/decision.hpp"
#include "branchscore/json.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace branchscore::systemone {

inline constexpr std::size_t max_questions_per_request = 16;
inline constexpr const char * default_model_id = "branchscore-local";

class RequestError : public std::runtime_error {
public:
    RequestError(std::string code, std::string message);
    const std::string & code() const noexcept;

private:
    std::string code_;
};

struct ChoiceQuestion {
    std::string id;
    DecisionRequest decision;
};

struct Request {
    std::string model_id;
    std::optional<std::string> request_id;
    std::vector<ChoiceQuestion> questions;
};

Request parse_request(
    const json::Value & envelope,
    const std::string & advertised_model_id = default_model_id);

json::Value make_response(
    const Request & request,
    const std::vector<DecisionResult> & results,
    bool include_raw_logits = true);

} // namespace branchscore::systemone
