#include "branchscore/systemone_adapter.hpp"

#include <cmath>
#include <limits>
#include <set>
#include <utility>

namespace branchscore::systemone {
namespace {

using Json = json::Value;
using Object = Json::Object;

const Json & required(const Json & object, const std::string & key) {
    const auto * value = object.find(key);
    if (value == nullptr) {
        throw RequestError("invalid_request", "missing required field: " + key);
    }
    return *value;
}

const std::string & required_string(const Json & object, const std::string & key) {
    const auto & value = required(object, key);
    if (value.type() != Json::Type::string) {
        throw RequestError("invalid_request", "field must be a string: " + key);
    }
    return value.string();
}

void require_object(const Json & value, const std::string & field) {
    if (value.type() != Json::Type::object) {
        throw RequestError("invalid_request", "field must be an object: " + field);
    }
}

void reject_unknown_fields(
    const Json & value,
    const std::set<std::string> & supported,
    const std::string & field) {
    for (const auto & [key, unused] : value.object()) {
        static_cast<void>(unused);
        if (supported.count(key) == 0) {
            throw RequestError("invalid_request", "unsupported field: " + field + "." + key);
        }
    }
}

std::string serialize_state(const Json & state) {
    switch (state.type()) {
        case Json::Type::string:
            if (state.string().empty()) {
                throw RequestError("invalid_request", "state must not be empty");
            }
            return state.string();
        case Json::Type::array:
        case Json::Type::object:
            return json::stringify(state);
        case Json::Type::null_value:
        case Json::Type::boolean:
        case Json::Type::number:
            throw RequestError(
                "invalid_request", "state must be a string, object, or array");
    }
    throw RequestError("invalid_request", "state has an unsupported JSON type");
}

Json number(const double value) {
    if (!std::isfinite(value)) {
        throw std::runtime_error("cannot encode a non-finite decision result");
    }
    return Json(value);
}

Json size_number(const std::size_t value) {
    return number(static_cast<double>(value));
}

Json timing_json(const TimingInfo & timing) {
    Object result;
    result.emplace("prompt_rendering_ms", number(timing.prompt_rendering_ms));
    result.emplace("tokenization_ms", number(timing.tokenization_ms));
    result.emplace("image_preprocessing_ms", number(timing.image_preprocessing_ms));
    result.emplace("vision_ms", number(timing.vision_ms));
    result.emplace("vision_backend_copy_ms", number(timing.vision_backend_copy_ms));
    result.emplace("vision_synchronization_ms", number(timing.vision_synchronization_ms));
    result.emplace("vision_graph_node_count", size_number(timing.vision_graph_node_count));
    result.emplace("prefill_ms", number(timing.prefill_ms));
    result.emplace("prefill_backend_copy_ms", number(timing.prefill_backend_copy_ms));
    result.emplace("prefill_synchronization_ms", number(timing.prefill_synchronization_ms));
    result.emplace("prefill_graph_node_count", size_number(timing.prefill_graph_node_count));
    result.emplace("readout_ms", number(timing.readout_ms));
    result.emplace("readout_backend_copy_ms", number(timing.readout_backend_copy_ms));
    result.emplace("readout_synchronization_ms", number(timing.readout_synchronization_ms));
    result.emplace("readout_graph_node_count", size_number(timing.readout_graph_node_count));
    result.emplace("normalization_ms", number(timing.normalization_ms));
    result.emplace("request_total_ms", number(timing.request_total_ms));
    return Json(std::move(result));
}

void validate_result(const ChoiceQuestion & question, const DecisionResult & result) {
    if (result.option_scores.size() != question.decision.options.size()) {
        throw std::runtime_error("decision result option count does not match request");
    }
    std::set<std::string> expected;
    for (const auto & option : question.decision.options) expected.insert(option.id);
    std::set<std::string> actual;
    for (const auto & score : result.option_scores) actual.insert(score.option_id);
    if (actual != expected || actual.size() != result.option_scores.size()) {
        throw std::runtime_error("decision result option IDs do not match request");
    }
    if (expected.count(result.selected_id) == 0) {
        throw std::runtime_error("selected option ID is not present in request");
    }
}

} // namespace

RequestError::RequestError(std::string code, std::string message)
    : std::runtime_error(std::move(message)), code_(std::move(code)) {}

const std::string & RequestError::code() const noexcept { return code_; }

Request parse_request(
    const Json & envelope,
    const std::string & advertised_model_id) {
    require_object(envelope, "request");
    reject_unknown_fields(
        envelope, {"state", "model", "questions", "request_id"}, "request");

    if (advertised_model_id.empty()) {
        throw std::invalid_argument("advertised model ID must not be empty");
    }

    Request result;
    const auto & state_value = required(envelope, "state");
    const auto state = serialize_state(state_value);
    result.model_id = required_string(envelope, "model");
    if (result.model_id != advertised_model_id) {
        throw RequestError("unsupported_model", "requested model is not served here");
    }

    if (const auto * request_id = envelope.find("request_id")) {
        if (request_id->type() != Json::Type::string) {
            throw RequestError("invalid_request", "field must be a string: request_id");
        }
        result.request_id = request_id->string();
    }

    const auto & questions = required(envelope, "questions");
    require_object(questions, "questions");
    if (questions.object().empty()) {
        throw RequestError("invalid_request", "questions must not be empty");
    }
    if (questions.object().size() > max_questions_per_request) {
        throw RequestError("invalid_request", "request exceeds the 16 question limit");
    }

    result.questions.reserve(questions.object().size());
    // The JSON codec stores object keys in lexical order, so both question and
    // criteria traversal are deterministic without relying on wire order.
    for (const auto & [question_id, question_value] : questions.object()) {
        if (question_id.empty()) {
            throw RequestError("invalid_request", "question IDs must not be empty");
        }
        require_object(question_value, "questions." + question_id);
        reject_unknown_fields(
            question_value, {"type", "instructions", "criteria"},
            "questions." + question_id);

        const auto & type = required_string(question_value, "type");
        if (type != "choice") {
            throw RequestError(
                "unsupported_question_type",
                "only type 'choice' is supported");
        }
        const auto & instructions = required_string(question_value, "instructions");
        if (instructions.empty()) {
            throw RequestError("invalid_request", "instructions must not be empty");
        }

        const auto & criteria = required(question_value, "criteria");
        require_object(criteria, "questions." + question_id + ".criteria");
        if (criteria.object().size() < 2 || criteria.object().size() > 16) {
            throw RequestError(
                "invalid_request", "choice criteria must contain 2-16 options");
        }

        DecisionRequest decision;
        decision.state = state;
        decision.question = instructions;
        decision.options.reserve(criteria.object().size());
        for (const auto & [option_id, description] : criteria.object()) {
            if (option_id.empty()) {
                throw RequestError("invalid_request", "criteria keys must not be empty");
            }
            if (description.type() != Json::Type::null_value &&
                description.type() != Json::Type::string) {
                throw RequestError(
                    "invalid_request", "criteria values must be strings or null");
            }
            const auto visible_description = description.type() == Json::Type::null_value
                ? option_id
                : option_id + ": " + description.string();
            decision.options.push_back({option_id, visible_description});
        }
        result.questions.push_back({question_id, std::move(decision)});
    }
    return result;
}

Json make_response(
    const Request & request,
    const std::vector<DecisionResult> & results,
    const bool include_raw_logits) {
    if (results.size() != request.questions.size()) {
        throw std::runtime_error("decision result count does not match request");
    }

    Object answers;
    Object diagnostic_questions;
    std::size_t input_tokens = 0;
    for (std::size_t index = 0; index < request.questions.size(); ++index) {
        const auto & question = request.questions[index];
        const auto & result = results[index];
        validate_result(question, result);
        const auto token_count = result.rendered_prompt_token_ids.size();
        if (token_count > std::numeric_limits<std::size_t>::max() - input_tokens) {
            throw std::runtime_error("input token count overflow");
        }
        input_tokens += token_count;

        Object probabilities;
        Object raw_logits;
        for (const auto & score : result.option_scores) {
            probabilities.emplace(score.option_id, number(score.relative_probability));
            raw_logits.emplace(score.option_id, number(score.raw_score));
        }

        Object answer;
        answer.emplace("type", "choice");
        answer.emplace("choice", result.selected_id);
        answer.emplace("probabilities", Json(std::move(probabilities)));
        answer.emplace("confidence", number(1.0));
        answers.emplace(question.id, Json(std::move(answer)));

        Object diagnostic;
        diagnostic.emplace("scoring_basis", result.scoring_basis);
        diagnostic.emplace("readout_id", result.readout_id);
        diagnostic.emplace("prompt_identity", result.rendered_prompt_identity);
        diagnostic.emplace("timings_ms", timing_json(result.timings));
        if (include_raw_logits) {
            diagnostic.emplace("raw_logits", Json(std::move(raw_logits)));
        }
        diagnostic_questions.emplace(question.id, Json(std::move(diagnostic)));
    }

    Object usage;
    usage.emplace("input_tokens", size_number(input_tokens));
    usage.emplace("output_tokens", number(0.0));

    Object branchscore_metadata;
    branchscore_metadata.emplace("schema_version", number(2.0));
    branchscore_metadata.emplace("confidence_kind", "constant_placeholder");
    branchscore_metadata.emplace("questions", Json(std::move(diagnostic_questions)));

    Object response;
    response.emplace("model", request.model_id);
    response.emplace("answers", Json(std::move(answers)));
    response.emplace("usage", Json(std::move(usage)));
    response.emplace("branchscore", Json(std::move(branchscore_metadata)));
    if (request.request_id) response.emplace("request_id", *request.request_id);
    return Json(std::move(response));
}

} // namespace branchscore::systemone
