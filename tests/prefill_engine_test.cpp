#include "branchscore/backend_context.hpp"
#include "branchscore/categorical_readout.hpp"
#include "branchscore/image_preprocessor.hpp"
#include "branchscore/model_loader.hpp"
#include "branchscore/prefill_engine.hpp"
#include "branchscore/tokenizer.hpp"
#include "branchscore/vision_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

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

        const std::vector<branchscore::TokenId> chunk_boundary_tokens(513, tokens.front());
        auto chunked = engine.prefill(chunk_boundary_tokens, nullptr, {});
        const auto chunked_logits = chunked.download_logits();
        if (chunked.prefix_length() != chunk_boundary_tokens.size() ||
            chunked.cache().cursor() != chunk_boundary_tokens.size() ||
            chunked.graph_count() != 2 || chunked.token_microbatch_size() != 512 ||
            chunked.cache_buffer_bytes() == 0 ||
            chunked.peak_graph_buffer_bytes() == 0 ||
            !std::all_of(chunked_logits.begin(), chunked_logits.end(), [](float value) {
                return std::isfinite(value);
            })) {
            throw std::runtime_error("chunked Prefill metadata or logits are invalid");
        }
        const auto chunked_readout = branchscore::gather_categorical_logits(
            chunked, {answer_a.id, answer_b.id}, backend);
        const auto expected_selected = chunked_readout.raw_scores[0] >=
            chunked_readout.raw_scores[1] ? answer_a.id : answer_b.id;
        const auto actual_selected = chunked_logits[answer_a.id] >=
            chunked_logits[answer_b.id] ? answer_a.id : answer_b.id;
        if (expected_selected != actual_selected) {
            throw std::runtime_error("chunked categorical readout disagrees with final logits");
        }

        auto * boundary_key = chunked.cache().key(0);
        const bool half = boundary_key->type == GGML_TYPE_F16;
        if (half != (backend.device().family == "CUDA") ||
            (half && chunked.cache().capacity() % 256 != 0)) {
            throw std::runtime_error("Prefill cache type/padding does not match the backend");
        }
        std::vector<float> key_at_511(boundary_key->ne[0]);
        std::vector<float> key_at_512(boundary_key->ne[0]);
        auto cache_timing = branchscore::BackendTiming{};
        const auto read_key = [&](std::size_t position, std::vector<float> & values) {
            if (half) {
                std::vector<ggml_fp16_t> packed(values.size());
                backend.tensor_get_timed(boundary_key, packed.data(), position * boundary_key->nb[1],
                    packed.size() * sizeof(ggml_fp16_t), cache_timing);
                std::transform(packed.begin(), packed.end(), values.begin(), ggml_fp16_to_fp32);
            } else {
                backend.tensor_get_timed(boundary_key, values.data(), position * boundary_key->nb[1],
                    values.size() * sizeof(float), cache_timing);
            }
        };
        read_key(511, key_at_511);
        read_key(512, key_at_512);
        const auto key_delta = std::inner_product(
            key_at_511.begin(), key_at_511.end(), key_at_512.begin(), 0.0,
            std::plus<>(), [](float left, float right) {
                return std::abs(left - right);
            });
        if (!std::isfinite(key_delta) || key_delta <= 1e-4) {
            throw std::runtime_error("KV values do not reflect distinct absolute positions");
        }
        std::cout << "chunked prefill tokens=" << chunked.prefix_length()
                  << " graphs=" << chunked.graph_count()
                  << " cache_bytes=" << chunked.cache_buffer_bytes()
                  << " peak_graph_bytes=" << chunked.peak_graph_buffer_bytes() << '\n';

        const std::vector<branchscore::TokenId> before_image(470, tokens.front());
        const std::vector<branchscore::TokenId> after_image(80, tokens.back());
        auto image_boundary = engine.prefill(before_image, &visual_tokens, after_image);
        const auto image_boundary_logits = image_boundary.download_logits();
        if (image_boundary.prefix_length() !=
                before_image.size() + visual_tokens.token_count() + after_image.size() ||
            image_boundary.graph_count() != 2 ||
            !std::all_of(
                image_boundary_logits.begin(), image_boundary_logits.end(),
                [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error("image-spanning chunk Prefill is invalid");
        }
        const auto image_boundary_readout = branchscore::gather_categorical_logits(
            image_boundary, {answer_a.id, answer_b.id}, backend);
        if (std::abs(
                image_boundary_readout.raw_scores[0] - image_boundary_logits[answer_a.id]) >
                1e-4F ||
            std::abs(
                image_boundary_readout.raw_scores[1] - image_boundary_logits[answer_b.id]) >
                1e-4F) {
            throw std::runtime_error(
                "image-spanning categorical readout disagrees with final logits");
        }
        std::cout << "image boundary prefill tokens=" << image_boundary.prefix_length()
                  << " graphs=" << image_boundary.graph_count() << '\n';

        std::cout << "categorical logits=" << categorical.raw_scores.size()
                  << " prefix_tokens=" << state.prefix_length() << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
