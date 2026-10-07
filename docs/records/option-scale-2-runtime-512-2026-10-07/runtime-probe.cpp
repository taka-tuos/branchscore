// Focused admission/resource/repetition probe through the production engine.
#include "branchscore/gemma4_decision_engine.hpp"
#include "branchscore/gemma4_prompt_renderer.hpp"
#include "branchscore/image_preprocessor.hpp"
#include "branchscore/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>

using namespace branchscore;
using J = json::Value;
int main(int argc, char ** argv) { try {
    if (argc != 5) throw std::runtime_error("MODEL MMPROJ INPUT_JSONL OUTPUT_JSONL");
    BackendContext backend("cuda");
    auto model = ModelLoader::load(argv[1], argv[2], backend);
    auto tokenizer = GemmaTokenizer::from_gguf(argv[1]);
    Gemma4DecisionEngine engine(model, backend, GemmaTokenizer::from_gguf(argv[1]));
    std::ifstream input(argv[3]);
    std::ofstream output(argv[4]), effective(std::string(argv[4])+".effective-inputs.jsonl");
    if (!input || !output || !effective) throw std::runtime_error("cannot open probe files");
    std::string line;
    while (std::getline(input, line)) {
        auto row = json::parse(line);
        DecisionRequest request;
        request.state = row.find("state")->string();
        request.question = row.find("question")->string();
        for (const auto & option : row.find("options")->array()) {
            request.options.push_back({option.find("id")->string(),option.find("description")->string()});
        }
        if (const auto * image = row.find("image")) request.image_path = image->string();
        const auto positions = [&] {
            const auto rendered = Gemma4PromptRenderer{}.render(request.state,request.question,request.options,
                request.has_image(),{}, {}, false);
            const auto count = tokenizer.tokenize(rendered.text,false,true).size();
            return request.has_image() ? count - 1 + ImagePreprocessor::load(*request.image_path,model.vision_config()).visual_token_count(model.vision_config()) : count;
        };
        if (const auto * fill = row.find("fill_positions")) {
            const auto target = std::size_t(fill->number());
            const auto base = request.state;
            std::size_t words = target - positions();
            for (int attempt = 0; attempt < 5; ++attempt) {
                request.state = base;
                for (std::size_t i=0;i<words;++i) request.state += " word";
                const auto count = positions();
                if (count==target) break;
                if (count<target) words += target-count; else words -= count-target;
            }
            if (positions()!=target) throw std::runtime_error("cannot prepare exact position boundary");
            row.set("state",J(request.state));
        }
        effective << json::stringify(row) << '\n';
        J::Object result;
        result.emplace("id",row.find("id")->string());
        result.emplace("option_count",J(double(request.options.size())));
        const auto started = std::chrono::steady_clock::now();
        try {
            const auto estimate = engine.estimate_prefill_positions(request);
            auto decision = engine.evaluate(request);
            if (row.find("expected_error")) throw std::runtime_error("oversized request was accepted");
            if (decision.option_scores.size()!=request.options.size() || estimate!=decision.timings.prefill_positions)
                throw std::runtime_error("score count/position estimate mismatch");
            std::set<TokenId> answer_ids;
            J::Array scores;
            double probability_sum=0;
            for (std::size_t i=0;i<decision.option_scores.size();++i) {
                const auto & score=decision.option_scores[i];
                if (score.option_id!=request.options[i].id || score.input_index!=i ||
                    !answer_ids.insert(score.answer_token_id).second || !std::isfinite(score.raw_score) ||
                    !std::isfinite(score.relative_probability)) throw std::runtime_error("invalid 512 score mapping");
                J::Object item;
                item.emplace("id",score.option_id); item.emplace("label",score.answer_label);
                item.emplace("token_id",J(double(score.answer_token_id)));
                item.emplace("raw_score",J(score.raw_score)); item.emplace("probability",J(score.relative_probability));
                scores.emplace_back(std::move(item)); probability_sum+=score.relative_probability;
            }
            if (std::abs(probability_sum-1)>1e-6) throw std::runtime_error("invalid probability sum");
            const auto best=std::max_element(decision.option_scores.begin(),decision.option_scores.end(),
                [](const auto & a,const auto & b){return a.raw_score<b.raw_score;});
            if (best->option_id!=decision.selected_id || best->input_index!=decision.selected_index)
                throw std::runtime_error("selected semantic ID does not match scores");
            result.emplace("scores",J(std::move(scores)));
            result.emplace("selected_id",decision.selected_id);
            result.emplace("expected_id",row.find("expected_selected_id")->string());
            result.emplace("renderer_id",decision.prompt_format.renderer_id);
            result.emplace("prompt_identity",decision.rendered_prompt_identity);
            const auto & timing=decision.timings;
            result.emplace("positions",J(double(timing.prefill_positions)));
            result.emplace("rendered_tokens",J(double(decision.rendered_prompt_token_ids.size())));
            result.emplace("visual_tokens",J(double(timing.prefill_positions-decision.rendered_prompt_token_ids.size()+(request.has_image()?1:0))));
            result.emplace("attention",timing.prefill_attention_path); result.emplace("kv_type",timing.prefill_kv_type);
            result.emplace("cache_bytes",J(double(timing.prefill_cache_bytes)));
            result.emplace("graph_bytes",J(double(timing.prefill_peak_graph_bytes)));
            result.emplace("graph_count",J(double(timing.prefill_graph_count)));
            result.emplace("tokenization_ms",J(timing.tokenization_ms)); result.emplace("vision_ms",J(timing.vision_ms));
            result.emplace("prefill_ms",J(timing.prefill_ms)); result.emplace("request_ms",J(timing.request_total_ms));
            std::cout << row.find("id")->string() << " positions=" << timing.prefill_positions
                      << " selected=" << decision.selected_id << " ms=" << timing.request_total_ms << std::endl;
        } catch (const DecisionBudgetError & error) {
            if (!row.find("expected_error") || row.find("expected_error")->string()!=error.code()) throw;
            result.emplace("error",error.code());
            std::cout << row.find("id")->string() << " rejected=" << error.code() << std::endl;
        }
        backend.synchronize();
        size_t available=0,total=0;
        ggml_backend_dev_memory(ggml_backend_get_device(backend.backend()),&available,&total);
        result.emplace("after_request_free_bytes",J(double(available)));
        result.emplace("elapsed_ms",J(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()));
        output << json::stringify(J(std::move(result))) << '\n'; output.flush();
    }
    return 0;
} catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; } }
