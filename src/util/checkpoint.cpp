#include "util/checkpoint.h"
#include <torch/serialize.h>
#include <fstream>

void save_checkpoint(const std::string& path, torch::nn::Module& model, torch::optim::Optimizer& opt, int epoch) {
  torch::serialize::OutputArchive archive;
  model.save(archive);
  opt.save(archive);
  archive.write("epoch", torch::tensor(epoch, torch::kInt32));
  archive.save_to(path);
}

bool load_checkpoint(const std::string& path, torch::nn::Module& model, torch::optim::Optimizer& opt, int& epoch_out) {
  std::ifstream f(path);
  if (!f.good()) return false;

  torch::serialize::InputArchive archive;
  archive.load_from(path);
  model.load(archive);
  opt.load(archive);

  torch::Tensor epoch_tensor;
  archive.read("epoch", epoch_tensor);
  epoch_out = epoch_tensor.item<int>();
  return true;
}
