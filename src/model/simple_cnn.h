#pragma once
#include <torch/torch.h>

struct SimpleCnnImpl : torch::nn::Module {
  SimpleCnnImpl();

  torch::Tensor forward(torch::Tensor x);

  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::Dropout dropout{nullptr};
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};
TORCH_MODULE(SimpleCnn);
