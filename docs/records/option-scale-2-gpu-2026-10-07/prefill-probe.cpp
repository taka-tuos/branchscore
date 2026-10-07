// Measurement harness only: production renderer/API still accepts 2-16 A-P options.
#include "branchscore/model_loader.hpp"
#include "branchscore/prefill_engine.hpp"
#include "branchscore/vision_encoder.hpp"
#include "branchscore/categorical_readout.hpp"
#include "branchscore/json.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <set>
#include <stdexcept>
using namespace branchscore;
using J = json::Value;
static std::string read(const std::string & p) {
    std::ifstream f(p); if (!f) throw std::runtime_error("cannot read " + p);
    return {std::istreambuf_iterator<char>(f), {}};
}
static void save_ids(const std::string & p, const std::vector<TokenId> & v) {
    std::ofstream f(p); for (auto x : v) f << x << '\n';
    if (!f) throw std::runtime_error("cannot write ids");
}
int main(int argc, char ** argv) { try {
    if (argc != 7) throw std::runtime_error("MODEL MMPROJ PROMPT LABEL_LINES IMAGE_OR_DASH OUTPUT_PREFIX");
    const std::string base = argv[6];
    BackendContext backend("cuda");
    J::Object result;
    auto memory = [&](const char * stage) {
        size_t available = 0, total = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(backend.backend()), &available, &total);
        result.emplace(std::string(stage) + "_free_bytes", J(double(available)));
        result.emplace("device_total_bytes", J(double(total)));
        std::cout << stage << " free_bytes=" << available << std::endl;
    };
    memory("before_load");
    auto model = ModelLoader::load(argv[1], argv[2], backend);
    result.emplace("text_weight_bytes", J(double(model.text_weight_bytes())));
    result.emplace("vision_weight_bytes", J(double(model.vision_weight_bytes())));
    memory("after_load");
    auto tokenizer = GemmaTokenizer::from_gguf(argv[1]);
    const auto prompt = read(argv[3]);
    const auto tokens = tokenizer.tokenize(prompt, false, true);
    std::vector<TokenId> answers; J::Array labels; std::set<TokenId> unique;
    std::ifstream label_file(argv[4]); std::string label;
    if (!label_file) throw std::runtime_error("cannot read labels");
    while (label_file >> label) {
        auto one = tokenizer.tokenize(label, false, true);
        auto combined = tokenizer.tokenize(prompt + label, false, true);
        if (one.size() != 1 || tokenizer.piece(one[0]) != label ||
            tokenizer.is_special_token(one[0]) || !unique.insert(one[0]).second ||
            combined.size() != tokens.size() + 1 || combined.back() != one[0] ||
            !std::equal(tokens.begin(), tokens.end(), combined.begin()))
            throw std::runtime_error("label boundary/uniqueness failed: " + label);
        answers.push_back(one[0]); labels.emplace_back(label);
    }
    if (answers.empty()) throw std::runtime_error("empty labels");
    save_ids(base + ".answers", answers);
    std::vector<TokenId> before = tokens, after;
    std::optional<VisualTokens> visual;
    if (std::string(argv[5]) != "-") {
        const auto marker = tokenizer.find_token("<|image|>");
        if (!marker) throw std::runtime_error("missing image marker");
        auto it = std::find(tokens.begin(), tokens.end(), *marker);
        if (it == tokens.end() || it == tokens.begin() || it + 1 == tokens.end() ||
            tokenizer.piece(*(it-1)) != "<|image>" || tokenizer.piece(*(it+1)) != "<image|>")
            throw std::runtime_error("invalid image boundaries");
        before.assign(tokens.begin(), it); after.assign(it + 1, tokens.end());
        VisionEncoder encoder(model, backend);
        visual.emplace(encoder.encode(ImagePreprocessor::load(argv[5], model.vision_config())));
        auto values = visual->download();
        std::ofstream f(base + ".embeddings", std::ios::binary);
        f.write(reinterpret_cast<const char *>(values.data()), values.size()*sizeof(float));
        if (!f) throw std::runtime_error("cannot write embeddings");
        result.emplace("vision_attention_path", J(vision_attention_path_name(visual->attention_path())));
    }
    save_ids(base + ".before", before); save_ids(base + ".after", after);
    memory("after_vision");
    PrefillEngine engine(model, backend);
    const auto started = std::chrono::steady_clock::now();
    auto state = engine.prefill(before, visual ? &*visual : nullptr, after);
    result.emplace("prefill_ms", J(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()));
    memory("after_prefill");
    const auto scores = gather_categorical_logits(state, answers, backend);
    const auto summary = summarize_categorical(scores.raw_scores);
    J::Array raw, probabilities;
    for (auto x : scores.raw_scores) raw.emplace_back(double(x));
    for (auto x : summary.relative_probabilities) probabilities.emplace_back(double(x));
    result.emplace("labels", J(std::move(labels))); result.emplace("logits", J(std::move(raw)));
    result.emplace("probabilities", J(std::move(probabilities)));
    result.emplace("selected_index", J(double(summary.selected_index)));
    result.emplace("positions", J(double(state.prefix_length())));
    result.emplace("visual_tokens", J(double(visual ? visual->token_count() : 0)));
    result.emplace("chunks", J(double(state.graph_count())));
    result.emplace("microbatch", J(double(state.token_microbatch_size())));
    result.emplace("cache_bytes", J(double(state.cache_buffer_bytes())));
    result.emplace("peak_graph_bytes", J(double(state.peak_graph_buffer_bytes())));
    #ifdef PROBE_FLASH
    result.emplace("path", J("F16 KV/mask, Flash with F32 precision request, full SWA, 256-position padding"));
#else
    result.emplace("path", J("F32 KV/mask, ordinary attention, full SWA"));
#endif
    std::ofstream f(base + ".json"); f << json::stringify(J(std::move(result))) << '\n';
    if (!f) throw std::runtime_error("cannot write output");
    return 0;
} catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; } }
