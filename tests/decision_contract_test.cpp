#include "branchscore/decision.hpp"

#include <iostream>
#include <stdexcept>

int main() {
    try {
        branchscore::DecisionRequest request;
        request.state = "state";
        request.question = "question";
        request.options = {{"semantic-a", "first"}, {"semantic-b", "second"}};
        request.chat_template_file = "/missing/template.jinja";

        branchscore::DecisionResult result;
        result.option_scores.resize(request.options.size());
        result.prompt_format.prompt_policy = request.prompt_policy;
        result.scoring_basis = "sum_logprob";
        result.terminator_scored = false;
        if (request.options.size() != 2 || request.options[0].id != "semantic-a" ||
            !request.chat_template_file || result.scoring_basis != "sum_logprob" ||
            result.terminator_scored ||
            result.prompt_format.prompt_policy.reasoning !=
                branchscore::ReasoningPolicy::DirectAnswerDisabled) {
            throw std::runtime_error("decision contract defaults are incorrect");
        }
        std::cout << "decision contract options=" << request.options.size() << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
