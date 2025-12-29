#include "train/trainer.h"
#include <iostream>

static double accuracy_from_logits(const torch::Tensor& logits, const torch::Tensor& y) {
  auto pred = logits.argmax(1);
  auto correct = pred.eq(y).sum().item<int64_t>();
  return static_cast<double>(correct) / static_cast<double>(y.size(0));
}

EpochResult train_one_epoch(torch::nn::Module& model,
                           torch::data::DataLoader<torch::data::Example<>>& loader,
                           torch::optim::Optimizer& opt,
                           torch::Device device,
                           int log_every) {
  model.train();
  EpochResult r{};
  int64_t seen = 0;
  double loss_sum = 0.0;
  double acc_sum = 0.0;

  int step = 0;
  for (auto& batch : loader) {
    auto x = batch.data.to(device);
    auto y = batch.target.to(device);

    opt.zero_grad();
    auto logits = model.forward(x);
    auto loss = torch::nn::functional::cross_entropy(logits, y);
    loss.backward();
    opt.step();

    const auto bs = y.size(0);
    seen += bs;
    loss_sum += loss.item<double>() * bs;
    acc_sum += accuracy_from_logits(logits.detach(), y.detach()) * bs;

    if (++step % log_every == 0) {
      std::cout << "step " << step
                << " loss=" << (loss_sum / seen)
                << " acc=" << (acc_sum / seen) << "\n";
    }
  }

  r.loss = loss_sum / seen;
  r.acc = acc_sum / seen;
  return r;
}

EpochResult eval_one_epoch(torch::nn::Module& model,
                          torch::data::DataLoader<torch::data::Example<>>& loader,
                          torch::Device device) {
  model.eval();
  torch::NoGradGuard no_grad;

  EpochResult r{};
  int64_t seen = 0;
  double loss_sum = 0.0;
  double acc_sum = 0.0;

  for (auto& batch : loader) {
    auto x = batch.data.to(device);
    auto y = batch.target.to(device);

    auto logits = model.forward(x);
    auto loss = torch::nn::functional::cross_entropy(logits, y);

    const auto bs = y.size(0);
    seen += bs;
    loss_sum += loss.item<double>() * bs;
    acc_sum += accuracy_from_logits(logits, y) * bs;
  }

  r.loss = loss_sum / seen;
  r.acc = acc_sum / seen;
  return r;
}
