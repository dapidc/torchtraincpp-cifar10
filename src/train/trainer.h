#pragma once
#include <torch/torch.h>

struct EpochResult {
  double loss = 0.0;
  double acc  = 0.0;
};

static inline double accuracy_from_logits(const torch::Tensor& logits, const torch::Tensor& y) {
  auto pred = logits.argmax(1);
  return pred.eq(y).to(torch::kFloat32).mean().item<double>();
}

template <typename Model, typename DataLoader>
EpochResult train_one_epoch(Model& model, DataLoader& loader, torch::Device device,
                           torch::optim::Optimizer& opt, int log_every) {
  model.train();
  double loss_sum = 0.0, acc_sum = 0.0;
  int64_t n = 0;
  int step = 0;

  for (auto& batch : loader) {
    auto x = batch.data.to(device);
    auto y = batch.target.to(device);

    opt.zero_grad();
    auto logits = model.forward(x);
    auto loss = torch::nn::functional::cross_entropy(logits, y);
    loss.backward();
    opt.step();

    const auto bs = x.size(0);
    loss_sum += loss.template item<double>() * bs;
    acc_sum  += accuracy_from_logits(logits, y) * bs;
    n += bs;

    if (log_every > 0 && (++step % log_every == 0)) {
      // optional: print progress
    }
  }

  return {loss_sum / (double)n, acc_sum / (double)n};
}

template <typename Model, typename DataLoader>
EpochResult eval_one_epoch(Model& model, DataLoader& loader, torch::Device device) {
  model.eval();
  torch::NoGradGuard ng;

  double loss_sum = 0.0, acc_sum = 0.0;
  int64_t n = 0;

  for (auto& batch : loader) {
    auto x = batch.data.to(device);
    auto y = batch.target.to(device);

    auto logits = model.forward(x);
    auto loss = torch::nn::functional::cross_entropy(logits, y);

    const auto bs = x.size(0);
    loss_sum += loss.template item<double>() * bs;
    acc_sum  += accuracy_from_logits(logits, y) * bs;
    n += bs;
  }

  return {loss_sum / (double)n, acc_sum / (double)n};
}
