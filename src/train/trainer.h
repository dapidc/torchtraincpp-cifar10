#pragma once
#include <torch/torch.h>
#include <string>

struct TrainConfig {
  int epochs = 10;
  int batch_size = 128;
  double lr = 1e-3;
  bool use_cuda = false;
  int log_every = 100;
  std::string out_dir = "runs";
  std::string data_dir = "data";
  std::string resume_from = "";
};

struct EpochResult {
  double loss = 0.0;
  double acc = 0.0;
};

EpochResult train_one_epoch(torch::nn::Module& model,
                           torch::data::DataLoader<torch::data::Example<>>& loader,
                           torch::optim::Optimizer& opt,
                           torch::Device device,
                           int log_every);

EpochResult eval_one_epoch(torch::nn::Module& model,
                          torch::data::DataLoader<torch::data::Example<>>& loader,
                          torch::Device device);
