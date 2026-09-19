#include "branchscore/backend_context.hpp"
#include "branchscore/image_preprocessor.hpp"
#include "branchscore/model_loader.hpp"
#include "branchscore/option_scorer.hpp"
#include "branchscore/prefill_engine.hpp"
#include "branchscore/tokenizer.hpp"
#include "branchscore/vision_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

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
        auto tokens = tokenizer.tokenize("Hello", true, false);
        branchscore::PrefillEngine engine(model, backend);
        auto state = engine.prefill(tokens, nullptr, {}, 4);
        const auto logits = state.download_logits();
        if (state.prefix_length() != tokens.size() ||
            state.cache().cursor() != tokens.size()) {
            throw std::runtime_error("Prefill did not freeze the expected prefix");
        }
        if (logits.size() != model.text_config().vocabulary_size ||
            !std::all_of(logits.begin(), logits.end(), [](float value) {
                return std::isfinite(value);
            })) {
            throw std::runtime_error("Prefill produced invalid logits");
        }
        const auto maximum_logit = *std::max_element(logits.begin(), logits.end());
        const auto minimum_logit = *std::min_element(logits.begin(), logits.end());
        if (maximum_logit - minimum_logit < 1.0F) {
            throw std::runtime_error("Prefill logits are unexpectedly constant");
        }

        state.cache().advance(4);
        state.cache().reset_branch();
        if (state.cache().cursor() != tokens.size()) {
            throw std::runtime_error("State-cache branch reset failed");
        }
        std::cout << "prefill tokens=" << tokens.size()
                  << " logits=" << logits.size() << '\n';

        const std::uint8_t white_pixel[] = {255, 255, 255};
        const auto image = branchscore::ImagePreprocessor::preprocess_rgb(
            white_pixel, 1, 1, model.vision_config());
        branchscore::VisionEncoder vision(model, backend);
        auto visual_tokens = vision.encode(image);
        auto multimodal = engine.prefill(
            {tokens.front()}, &visual_tokens, {tokens.back()}, 4);
        const auto multimodal_logits = multimodal.download_logits();
        const auto expected_prefix = visual_tokens.token_count() + 2;
        if (multimodal.prefix_length() != expected_prefix ||
            multimodal_logits.size() != model.text_config().vocabulary_size ||
            !std::all_of(
                multimodal_logits.begin(), multimodal_logits.end(),
                [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error("Multimodal Prefill produced an invalid state");
        }
        std::cout << "multimodal prefill tokens=" << expected_prefix << '\n';

        branchscore::OptionTokens option;
        option.option_id = "first";
        option.input_index = 0;
        option.ids = {9259, 1902};
        option.boundary_valid = true;
        branchscore::OptionScorer scorer(model, backend);
        const auto expected_logits = state.download_logits();
        const auto expected_max = *std::max_element(
            expected_logits.begin(), expected_logits.end());
        double expected_normalizer = 0.0;
        for (const auto value : expected_logits) {
            expected_normalizer += std::exp(value - expected_max);
        }
        const auto expected_first =
            static_cast<double>(expected_logits[option.ids.front()] - expected_max) -
            std::log(expected_normalizer);
        const auto first_score = scorer.score(state, option);
        if (first_score.token_count != option.ids.size() ||
            first_score.token_logprobs.size() != option.ids.size() ||
            !std::isfinite(first_score.sum_logprob) ||
            !std::isfinite(first_score.mean_logprob) ||
            std::abs(first_score.token_logprobs.front() - expected_first) > 1e-4 ||
            state.cache().cursor() != state.prefix_length() + option.ids.size() - 1) {
            throw std::runtime_error("multi-token option scoring produced invalid state");
        }
        const auto second_score = scorer.score(state, option);
        if (std::abs(second_score.sum_logprob - first_score.sum_logprob) > 1e-4) {
            throw std::runtime_error("option branch reset is not deterministic");
        }
        std::cout << "option score tokens=" << first_score.token_count
                  << " sum=" << first_score.sum_logprob << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
