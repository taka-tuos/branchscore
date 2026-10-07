#include "branchscore/model_loader.hpp"
#include "branchscore/vision_encoder.hpp"
#include "branchscore/image_preprocessor.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char ** argv) {
 if (argc!=4) throw std::runtime_error("MODEL MMPROJ IMAGE_OUTPUT_LIST");
 branchscore::BackendContext backend("cpu");
 auto model=branchscore::ModelLoader::load(argv[1],argv[2],backend);
 branchscore::VisionEncoder encoder(model,backend);
 std::ifstream list(argv[3]);std::string image,output;
 while(list>>image>>output) {
  auto prepared=branchscore::ImagePreprocessor::load(image,model.vision_config());
  auto visual=encoder.encode(prepared);auto values=visual.download();
  std::ofstream f(output,std::ios::binary);f.write(reinterpret_cast<const char *>(values.data()),values.size()*sizeof(float));
  if (!f) throw std::runtime_error("embedding write failed");
  std::cout<<image<<" tokens="<<visual.token_count()<<" width="<<visual.embedding_length()<<std::endl;
 }
}
