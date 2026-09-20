#include "branchscore/backend_context.hpp"
#include "branchscore/categorical_readout.hpp"
#include "branchscore/image_preprocessor.hpp"
#include "branchscore/model_loader.hpp"
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
        auto state = engine.prefill(tokens, nullptr, {});
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

        const auto answer_a = tokenizer.tokenize_answer_label("Hello", "A");
        const auto answer_b = tokenizer.tokenize_answer_label("Hello", "B");
        const auto categorical = branchscore::gather_categorical_logits(
            state, {answer_a.id, answer_b.id}, backend);
        if (categorical.raw_scores.size() != 2 ||
            std::abs(categorical.raw_scores[0] - logits[answer_a.id]) > 1e-4F ||
            std::abs(categorical.raw_scores[1] - logits[answer_b.id]) > 1e-4F) {
            throw std::runtime_error("categorical gather disagrees with Prefill logits");
        }
        const auto categorical_summary =
            branchscore::summarize_categorical(categorical.raw_scores);
        if (categorical_summary.relative_probabilities.size() != 2 ||
            !std::isfinite(categorical_summary.relative_probabilities[0]) ||
            !std::isfinite(categorical_summary.relative_probabilities[1])) {
            throw std::runtime_error("categorical normalization failed");
        }
        std::cout << "prefill tokens=" << tokens.size()
                  << " logits=" << logits.size() << '\n';

        const std::uint8_t white_pixel[] = {255, 255, 255};
        const auto image = branchscore::ImagePreprocessor::preprocess_rgb(
            white_pixel, 1, 1, model.vision_config());
        branchscore::VisionEncoder vision(model, backend);
        auto visual_tokens = vision.encode(image);
        auto multimodal = engine.prefill(
            {tokens.front()}, &visual_tokens, {tokens.back()});
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

        std::cout << "categorical logits=" << categorical.raw_scores.size()
                  << " prefix_tokens=" << state.prefix_length() << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
