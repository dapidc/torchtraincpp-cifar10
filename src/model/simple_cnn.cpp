#include "model/simple_cnn.h"

SimpleCnnImpl::SimpleCnnImpl() {
  conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).padding(1)));
  conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
  dropout = register_module("dropout", torch::nn::Dropout(0.25));
  fc1 = register_module("fc1", torch::nn::Linear(64 * 8 * 8, 256));
  fc2 = register_module("fc2", torch::nn::Linear(256, 10));
}

torch::Tensor SimpleCnnImpl::forward(torch::Tensor x) {
  x = torch::relu(conv1->forward(x));
  x = torch::max_pool2d(x, 2); // 32x32 -> 16x16
  x = torch::relu(conv2->forward(x));
  x = torch::max_pool2d(x, 2); // 16x16 -> 8x8
  x = dropout->forward(x);
  x = x.view({x.size(0), -1});
  x = torch::relu(fc1->forward(x));
  x = dropout->forward(x);
  x = fc2->forward(x);
  return x;
}
