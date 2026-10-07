#include "llama.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<int> read_ids(const char * path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error(std::string("open failed: ") + path);
    std::vector<int> out;
    int value;
    while (in >> value) out.push_back(value);
    return out;
}

static void decode_tokens(llama_context * ctx, const std::vector<int> & ids, llama_pos & pos, bool output_last) {
    size_t offset = 0;
    while (offset < ids.size()) {
        const int32_t count = static_cast<int32_t>(std::min<size_t>(512, ids.size() - offset));
        auto batch = llama_batch_init(count, 0, 1);
        batch.n_tokens = count;
        for (int32_t i = 0; i < count; ++i) {
            batch.token[i] = ids[offset + i];
            batch.pos[i] = pos++;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = output_last && offset + i + 1 == ids.size() ? 1 : 0;
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) throw std::runtime_error("llama_decode(text) failed: " + std::to_string(rc));
        offset += count;
    }
}

static void decode_embeddings(llama_context * ctx, const float * values, size_t token_count, int32_t embd, llama_pos & pos) {
    size_t offset = 0;
    while (offset < token_count) {
        const int32_t count = static_cast<int32_t>(std::min<size_t>(512, token_count - offset));
        auto batch = llama_batch_init(count, embd, 1);
        batch.n_tokens = count;
        std::copy(values + offset * embd, values + (offset + count) * embd, batch.embd);
        for (int32_t i = 0; i < count; ++i) {
            batch.pos[i] = pos++;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 0;
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) throw std::runtime_error("llama_decode(image embd) failed: " + std::to_string(rc));
        offset += count;
    }
}

int main(int argc, char ** argv) {
    try {
        if (argc != 8) throw std::runtime_error("usage: MODEL MMPROJ BEFORE_IDS AFTER_IDS ANSWER_IDS HEADERLESS_F32_EMBD_IN USE_EXTRA_BUFTS");
        const auto before = read_ids(argv[3]);
        const auto after = read_ids(argv[4]);
        const auto answers = read_ids(argv[5]);
        if (before.empty() || after.empty() || answers.empty()) throw std::runtime_error("empty ids");

        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.use_extra_bufts = std::string(argv[7]) == "1";
        ggml_backend_dev_t cpu_devices[] = {ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr};
        mp.devices = cpu_devices;
        auto * model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("text model load failed");

        if (!std::ifstream(argv[2], std::ios::binary)) throw std::runtime_error("mmproj path unavailable (record provenance check)");
        const int32_t embd = llama_model_n_embd_inp(model);
        if (embd <= 0) throw std::runtime_error("invalid model input embedding width");
        std::ifstream exact_image(argv[6], std::ios::binary | std::ios::ate);
        if (!exact_image) throw std::runtime_error("failed to open exact image embeddings");
        const auto embedding_bytes = exact_image.tellg();
        const auto bytes_per_token = static_cast<std::streamoff>(embd) * sizeof(float);
        if (embedding_bytes <= 0 || embedding_bytes % bytes_per_token != 0) {
            throw std::runtime_error("exact embedding byte count does not match model width");
        }
        const size_t image_tokens = static_cast<size_t>(embedding_bytes / bytes_per_token);
        std::vector<float> shared_image_embd(image_tokens * static_cast<size_t>(embd));
        exact_image.seekg(0);
        exact_image.read(reinterpret_cast<char *>(shared_image_embd.data()), embedding_bytes);
        if (!exact_image) throw std::runtime_error("failed to read exact image embeddings");
        const float * image_embd = shared_image_embd.data();

        const size_t full_positions = before.size() + image_tokens + after.size();
        auto cp = llama_context_default_params();
        cp.n_ctx = static_cast<uint32_t>(full_positions + 1);
        cp.n_batch = 512;
        cp.n_ubatch = 512;
        cp.n_seq_max = 1;
        cp.type_k = GGML_TYPE_F32;
        cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.swa_full = true;
        auto * ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("context creation failed");
        llama_pos pos = 0;
        decode_tokens(ctx, before, pos, false);
        decode_embeddings(ctx, image_embd, image_tokens, embd, pos);
        decode_tokens(ctx, after, pos, true);
        if (static_cast<size_t>(pos) != full_positions) throw std::runtime_error("position count mismatch");

        const auto * logits = llama_get_logits(ctx);
        if (!logits) throw std::runtime_error("no final logits");
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        std::cout << std::setprecision(9) << "before=" << before.size() << " image_tokens=" << image_tokens
                  << " after=" << after.size() << " positions=" << pos << " embedding_width=" << embd
                  << " answers=" << answers.size() << '\n';
        for (size_t i = 0; i < answers.size(); ++i) {
            const int id = answers[i];
            if (id < 0 || id >= n_vocab) throw std::runtime_error("answer token out of range");
            std::cout << "answer_" << i << " id=" << id << " logit=" << logits[id] << '\n';
        }
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
