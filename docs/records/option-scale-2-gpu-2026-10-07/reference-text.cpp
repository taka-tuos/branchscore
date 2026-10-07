// CUDA adaptation of the preserved Step 4 F32/ordinary/full-SWA reference.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 6) throw std::runtime_error("usage: llama-step1-reference MODEL IDS OUTPUT");
    std::ifstream input(argv[2]);
    if (!input) throw std::runtime_error("failed to open token IDs");
    std::vector<llama_token> tokens;
    std::int32_t value = 0;
    while (input >> value) tokens.push_back(value);
    if (tokens.empty()) throw std::runtime_error("token list is empty");

    ggml_backend_load_all();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    model_params.use_extra_bufts = std::string(argv[5]) == "1";
    ggml_backend_dev_t cuda_devices[] = {
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU), nullptr,
    };
    model_params.devices = cuda_devices;
    llama_model_tensor_buft_override overrides[] = {
        {".*", ggml_backend_dev_buffer_type(cuda_devices[0])}, {nullptr, nullptr},
    };
    model_params.tensor_buft_overrides = overrides;
    auto * model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr) throw std::runtime_error("failed to load model");

    auto context_params = llama_context_default_params();
    context_params.n_ctx = static_cast<std::uint32_t>(tokens.size() + 1);
    context_params.n_batch = 512;
    context_params.n_ubatch = 512;
    context_params.type_k = GGML_TYPE_F32;
    context_params.type_v = GGML_TYPE_F32;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    context_params.swa_full = true;
    auto * context = llama_init_from_model(model, context_params);
    if (context == nullptr) throw std::runtime_error("failed to create reference context");
    auto batch = llama_batch_init(512, 0, 1);
    for (size_t offset = 0; offset < tokens.size(); offset += 512) {
        const size_t count = std::min<size_t>(512, tokens.size() - offset);
        batch.n_tokens = count;
        for (size_t i = 0; i < count; ++i) {
            batch.token[i] = tokens[offset+i]; batch.pos[i] = offset+i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
            batch.logits[i] = std::string(argv[4]) == "1" || offset+i+1 == tokens.size();
        }
        if (llama_decode(context, batch) != 0) {
            throw std::runtime_error("reference decode failed");
        }
    }

    const auto * logits = llama_get_logits_ith(context, -1);
    const auto * vocab = llama_model_get_vocab(model);
    const auto vocab_size = llama_vocab_n_tokens(vocab);
    std::ofstream output(argv[3], std::ios::binary);
    if (!output) throw std::runtime_error("failed to open output logits file");
    output.write(reinterpret_cast<const char *>(logits),
                 static_cast<std::streamsize>(vocab_size) * sizeof(float));
    if (!output) throw std::runtime_error("failed to write output logits file");
    std::cout << "tokens=" << tokens.size() << " vocab=" << vocab_size << '\n';
    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
    return 0;
}
