#include "branchscore/model_loader.hpp"
#include "branchscore/vision_encoder.hpp"
#include "branchscore/prefill_engine.hpp"
#include "ggml-cpu.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdexcept>
static std::vector<int> ids(const char *p) { std::ifstream f(p); if(!f) throw std::runtime_error(p); std::vector<int> v; int x; while(f>>x)v.push_back(x); return v; }
int main(int argc,char **argv) { try {
 if(argc!=9) throw std::runtime_error("MODEL MMPROJ BEFORE AFTER ANSWERS EMBEDDINGS IMAGE MODE");
 branchscore::BackendContext backend("cpu"); ggml_backend_cpu_set_n_threads(backend.backend(),4);
 auto model=branchscore::ModelLoader::load(argv[1],argv[2],backend);
 auto before=ids(argv[3]),after=ids(argv[4]),answers=ids(argv[5]);
 auto image=branchscore::ImagePreprocessor::load(argv[7],model.vision_config());
 branchscore::VisionEncoder encoder(model,backend); auto visual=encoder.encode(image);
 std::vector<float> shared(visual.token_count()*visual.embedding_length());
 std::ifstream f(argv[6],std::ios::binary); f.read(reinterpret_cast<char*>(shared.data()),shared.size()*sizeof(float));
 if(!f || f.peek()!=EOF) throw std::runtime_error("embedding size mismatch");
 ggml_backend_tensor_set(visual.tensor(),shared.data(),0,shared.size()*sizeof(float));
 branchscore::PrefillEngine engine(model,backend); auto start=std::chrono::steady_clock::now();
 auto state=engine.prefill(before,&visual,after); auto end=std::chrono::steady_clock::now(); auto logits=state.download_logits();
 std::cout<<std::setprecision(9)<<"positions="<<state.prefix_length()<<" cache_bytes="<<state.cache_buffer_bytes()<<" graph_bytes="<<state.peak_graph_buffer_bytes()<<" chunks="<<state.graph_count()<<" prefill_ms="<<std::chrono::duration<double,std::milli>(end-start).count()<<'\n';
 for(size_t i=0;i<answers.size();++i) std::cout<<"answer_"<<i<<" id="<<answers[i]<<" logit="<<logits.at(answers[i])<<'\n';
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;} }
