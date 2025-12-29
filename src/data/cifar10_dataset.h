#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>

class Cifar10Dataset : public torch::data::datasets::Dataset<Cifar10Dataset> {
public:
  // root should contain "cifar-10-batches-bin"
  Cifar10Dataset(std::string root, bool train);

  torch::data::Example<> get(size_t index) override;
  torch::optional<size_t> size() const override { return images_.size(); }

private:
  void load_file(const std::string& path);

  std::vector<torch::Tensor> images_;
  std::vector<int64_t> labels_;
};
