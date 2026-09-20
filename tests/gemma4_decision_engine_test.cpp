#include "branchscore/backend_context.hpp"
#include "branchscore/gemma4_decision_engine.hpp"
#include "branchscore/model_loader.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void check_result(const branchscore::DecisionResult & result, bool expect_requested_template) {
    if (result.option_scores.size() != 2 || result.selected_id.empty() ||
        result.selected_index >= result.option_scores.size() ||
        result.option_scores[result.selected_index].option_id != result.selected_id ||
        result.scoring_basis != "answer_slot_logit" ||
        result.readout_id != "gemma4-next-token-categorical-v1" ||
        result.terminator_scored ||
        result.prompt_format.renderer_id != "gemma4-categorical-v1" ||
        result.prompt_format.effective_source != "built-in" ||
        result.prompt_format.requested_template_applied ||
        result.prompt_format.requested_template_file.has_value() != expect_requested_template ||
        result.prompt_format.gguf_chat_template_used ||
        result.rendered_prompt_identity.find("gemma4-categorical-v1/sha256:") != 0 ||
        result.rendered_prefix_token_ids.empty() ||
        result.timings.readout_graph_node_count == 0) {
        throw std::runtime_error("decision result contract is incorrect");
    }
    const auto nonnegative = [](double value) {
        return std::isfinite(value) && value >= 0.0;
    };
    if (!nonnegative(result.timings.prompt_rendering_ms) ||
        !nonnegative(result.timings.tokenization_ms) ||
        !nonnegative(result.timings.image_preprocessing_ms) ||
        !nonnegative(result.timings.vision_ms) ||
        !nonnegative(result.timings.prefill_ms) ||
        !nonnegative(result.timings.readout_ms) ||
        !nonnegative(result.timings.score_total_ms) ||
        !nonnegative(result.timings.normalization_ms) ||
        !nonnegative(result.timings.request_total_ms)) {
        throw std::runtime_error("decision timing fields are invalid");
    }
    double probability_sum = 0.0;
    for (const auto & score : result.option_scores) {
        if (score.input_index >= result.option_scores.size() ||
            score.answer_label.size() != 1 || score.answer_token_id < 0 ||
            !std::isfinite(score.raw_score) ||
            !std::isfinite(score.relative_probability) || score.relative_probability < 0.0 ||
            score.relative_probability > 1.0) {
            throw std::runtime_error("option result contract is incorrect");
        }
        probability_sum += score.relative_probability;
    }
    if (std::abs(probability_sum - 1.0) > 1e-6) {
        throw std::runtime_error("relative probabilities are not normalized");
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 4 || std::string(argv[1]).empty() || std::string(argv[2]).empty()) {
        std::cerr << "skipping: BRANCHSCORE_TEST_MODEL and "
                     "BRANCHSCORE_TEST_MMPROJ are not set\n";
        return 77;
    }

    try {
        branchscore::BackendContext backend(argv[3]);
        auto model = branchscore::ModelLoader::load(argv[1], argv[2], backend);
        auto tokenizer = branchscore::GemmaTokenizer::from_gguf(argv[1]);
        branchscore::Gemma4DecisionEngine engine(
            model, backend, std::move(tokenizer));

        branchscore::DecisionRequest request;
        request.state = "The service is healthy.";
        request.question = "Which action should be taken?";
        request.options = {
            {"keep", "Keep it running"},
            {"stop", "Stop it"},
        };
        auto invalid = request;
        invalid.options.pop_back();
        bool rejected_invalid_count = false;
        try {
            (void) engine.evaluate(invalid);
        } catch (const std::runtime_error &) {
            rejected_invalid_count = true;
        }
        if (!rejected_invalid_count) {
            throw std::runtime_error("engine accepted an invalid option count");
        }
        request.chat_template_file = "/definitely/nonexistent/template.jinja";
        const auto no_op = engine.evaluate(request);
        check_result(no_op, true);

        request.chat_template_file.reset();
        const auto baseline = engine.evaluate(request);
        check_result(baseline, false);
        if (no_op.rendered_prefix_token_ids != baseline.rendered_prefix_token_ids ||
            no_op.rendered_prompt_identity != baseline.rendered_prompt_identity ||
            no_op.selected_id != baseline.selected_id ||
            no_op.option_scores.size() != baseline.option_scores.size()) {
            throw std::runtime_error(
                "reserved template override changed the decision path");
        }
        for (std::size_t i = 0; i < no_op.option_scores.size(); ++i) {
            if (std::abs(no_op.option_scores[i].raw_score -
                        baseline.option_scores[i].raw_score) > 1e-4) {
                throw std::runtime_error(
                    "reserved template override changed option scores");
            }
        }
        std::cout << "engine selected=" << baseline.selected_id
                  << " prefix_tokens=" << baseline.rendered_prefix_token_ids.size() << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
