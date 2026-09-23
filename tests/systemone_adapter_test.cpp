#include "branchscore/systemone_adapter.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = branchscore::json::Value;

void expect(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

branchscore::DecisionResult result_for(
    const branchscore::systemone::ChoiceQuestion & question,
    const std::string & selected,
    const std::size_t token_count) {
    branchscore::DecisionResult result;
    result.selected_id = selected;
    result.readout_id = "gemma4-next-token-categorical-v1";
    result.scoring_basis = "answer_slot_logit";
    result.rendered_prompt_identity = "test-prompt-identity";
    result.rendered_prompt_token_ids.resize(token_count);
    for (std::size_t index = 0; index < question.decision.options.size(); ++index) {
        const auto & option = question.decision.options[index];
        const bool chosen = option.id == selected;
        result.option_scores.push_back({
            index,
            option.id,
            index == 0 ? "A" : "B",
            static_cast<std::int32_t>(index + 10),
            chosen ? 2.5 : -1.0,
            chosen ? 0.95 : 0.05,
        });
    }
    result.timings.request_total_ms = 1.25;
    result.timings.prefill_ms = 1.0;
    return result;
}

void expect_error(const std::string & input, const std::string & expected_code) {
    bool matched = false;
    try {
        static_cast<void>(branchscore::systemone::parse_request(
            branchscore::json::parse(input)));
    } catch (const branchscore::systemone::RequestError & error) {
        matched = error.code() == expected_code;
    }
    expect(matched, "adapter did not return expected request error: " + expected_code);
}

} // namespace

int main() {
    try {
        const auto envelope = branchscore::json::parse(R"({
            "request_id":"adapter-check",
            "state":{"z":2,"a":1},
            "model":"branchscore-local",
            "questions":{
                "zeta":{"type":"choice","instructions":"Question Z?","criteria":{"yes":"Yes","no":null}},
                "alpha":{"type":"choice","instructions":"Question A?","criteria":{"up":null,"down":"Turn it off"}}
            }
        })");
        const auto request = branchscore::systemone::parse_request(envelope);
        expect(request.questions.size() == 2, "question count changed");
        expect(request.questions[0].id == "alpha" && request.questions[1].id == "zeta",
               "question map order is not deterministic lexical order");
        expect(request.questions[0].decision.state == R"({"a":1,"z":2})",
               "structured state was not serialized deterministically");
        expect(request.questions[0].decision.question == "Question A?",
               "question ID or instructions mapping changed");
        expect(request.questions[0].decision.options.size() == 2 &&
                   request.questions[0].decision.options[0].id == "down" &&
                   request.questions[0].decision.options[0].description == "down: Turn it off" &&
                   request.questions[0].decision.options[1].id == "up" &&
                   request.questions[0].decision.options[1].description == "up",
               "criteria IDs, order, or visible descriptions changed");
        expect(request.image_bytes == nullptr, "text request unexpectedly has an image");

        std::vector<branchscore::DecisionResult> results;
        results.push_back(result_for(request.questions[0], "up", 11));
        results.push_back(result_for(request.questions[1], "no", 13));
        const auto response = branchscore::systemone::make_response(request, results, true);
        const auto * echoed_id = response.find("request_id");
        const auto * answers = response.find("answers");
        const auto * usage = response.find("usage");
        const auto * metadata = response.find("branchscore");
        expect(echoed_id != nullptr && echoed_id->string() == "adapter-check",
               "request ID was not echoed");
        expect(answers != nullptr && answers->find("alpha") != nullptr &&
                   answers->find("zeta") != nullptr,
               "answers are not mapped by question ID");
        const auto * alpha = answers->find("alpha");
        expect(alpha->find("choice")->string() == "up" &&
                   std::fabs(alpha->find("confidence")->number() - 1.0) < 1e-12,
               "Choice answer projection changed");
        expect(alpha->find("probabilities")->find("up") != nullptr &&
                   std::fabs(alpha->find("probabilities")->find("up")->number() - 0.95) < 1e-12,
               "probability map lost an option ID or value");
        expect(usage != nullptr && usage->find("input_tokens")->number() == 24.0 &&
                   usage->find("output_tokens")->number() == 0.0,
               "usage token counts changed");
        expect(metadata != nullptr && metadata->find("schema_version")->number() == 2.0 &&
                   metadata->find("confidence_kind")->string() == "constant_placeholder",
               "branchscore metadata changed");
        const auto * diagnostic = metadata->find("questions")->find("alpha");
        expect(diagnostic != nullptr && diagnostic->find("raw_logits") != nullptr &&
                   diagnostic->find("prompt_identity") != nullptr &&
                   diagnostic->find("timings_ms") != nullptr,
               "per-question diagnostics are missing");
        expect(response.find("rendered_prompt_token_ids") == nullptr &&
                   response.find("prompt") == nullptr,
               "HTTP projection exposed prompt diagnostics");

        const auto without_logits = branchscore::systemone::make_response(request, results, false);
        const auto * compact_diagnostic = without_logits.find("branchscore")
            ->find("questions")->find("alpha");
        expect(compact_diagnostic->find("raw_logits") == nullptr,
               "raw logits were included when disabled");

        expect_error(
            R"({"state":"x","model":"branchscore-local","questions":{"q":{"type":"score","instructions":"x","criteria":["a","b"]}}})",
            "unsupported_question_type");
        expect_error(
            R"({"state":"x","model":"branchscore-local","questions":{"q":{"type":"noul","instructions":"x"}}})",
            "unsupported_question_type");
        expect_error(
            R"({"state":"x","model":"other","questions":{"q":{"type":"choice","instructions":"x","criteria":{"a":"A","b":"B"}}}})",
            "unsupported_model");
        expect_error(
            R"({"state":"x","model":"branchscore-local","image":{"media_type":"image/gif","data_base64":"ignored"},"questions":{"q":{"type":"choice","instructions":"x","criteria":{"a":"A","b":"B"}}}})",
            "unsupported_media_type");
        expect_error(
            R"({"state":"x","model":"branchscore-local","image":{"media_type":"image/png","data_base64":"iVBORw0KGgo="},"questions":{"q":{"type":"choice","instructions":"x","criteria":{"a":"A","b":"B"}}}})",
            "invalid_image");

        Json::Object questions;
        for (int index = 0; index < 17; ++index) {
            Json::Object criteria;
            criteria.emplace("a", Json("A"));
            criteria.emplace("b", Json("B"));
            Json::Object question;
            question.emplace("type", Json("choice"));
            question.emplace("instructions", Json("choose"));
            question.emplace("criteria", Json(std::move(criteria)));
            questions.emplace("q" + std::to_string(index), Json(std::move(question)));
        }
        Json::Object too_many;
        too_many.emplace("state", Json("x"));
        too_many.emplace("model", Json("branchscore-local"));
        too_many.emplace("questions", Json(std::move(questions)));
        bool question_limit_rejected = false;
        try {
            static_cast<void>(branchscore::systemone::parse_request(Json(std::move(too_many))));
        } catch (const branchscore::systemone::RequestError & error) {
            question_limit_rejected = error.code() == "invalid_request";
        }
        expect(question_limit_rejected, "question limit above 16 was accepted");

        std::cout << "systemone adapter checks passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "systemone adapter test: " << error.what() << '\n';
        return 1;
    }
}
