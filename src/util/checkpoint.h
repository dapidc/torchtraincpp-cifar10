#pragma once
#include <torch/torch.h>
#include <string>

void save_checkpoint(const std::string& path, torch::nn::Module& model, torch::optim::Optimizer& opt, int epoch);
bool load_checkpoint(const std::string& path, torch::nn::Module& model, torch::optim::Optimizer& opt, int& epoch_out);
