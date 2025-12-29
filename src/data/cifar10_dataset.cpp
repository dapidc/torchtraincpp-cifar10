#include "data/cifar10_dataset.h"
#include <fstream>
#include <stdexcept>

namespace {
constexpr int kImageH = 32;
constexpr int kImageW = 32;
constexpr int kChannels = 3;
constexpr int kImageBytes = kImageH * kImageW * kChannels; // 3072
constexpr int kRecordBytes = 1 + kImageBytes;              // label + image
} // namespace

Cifar10Dataset::Cifar10Dataset(std::string root, bool train) {
  const std::string base = root + "/cifar-10-batches-bin/";
  if (train) {
    for (int i = 1; i <= 5; i++) {
      load_file(base + "data_batch_" + std::to_string(i) + ".bin");
    }
  } else {
    load_file(base + "test_batch.bin");
  }

  if (images_.empty()) {
    throw std::runtime_error("No CIFAR-10 data loaded. Check dataset path: " + base);
  }
}

void Cifar10Dataset::load_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Failed to open: " + path);

  std::vector<unsigned char> buffer(kRecordBytes);

  while (f.read(reinterpret_cast<char*>(buffer.data()), kRecordBytes)) {
    const uint8_t label = buffer[0];

    // CIFAR binary layout: R(1024) + G(1024) + B(1024)
    auto img = torch::empty({kChannels, kImageH, kImageW}, torch::kUInt8);

    auto r = torch::from_blob(buffer.data() + 1, {kImageH * kImageW}, torch::kUInt8).clone();
    auto g = torch::from_blob(buffer.data() + 1 + 1024, {kImageH * kImageW}, torch::kUInt8).clone();
    auto b = torch::from_blob(buffer.data() + 1 + 2048, {kImageH * kImageW}, torch::kUInt8).clone();

    img[0] = r.view({kImageH, kImageW});
    img[1] = g.view({kImageH, kImageW});
    img[2] = b.view({kImageH, kImageW});

    // Convert to float in [0,1]
    img = img.to(torch::kFloat32).div_(255.0);

    images_.push_back(img);
    labels_.push_back(static_cast<int64_t>(label));
  }
}

torch::data::Example<> Cifar10Dataset::get(size_t index) {
  // Normalize using common CIFAR-10 stats
  // mean = (0.4914, 0.4822, 0.4465), std = (0.2470, 0.2435, 0.2616)
  auto x = images_.at(index).clone();
  auto y = torch::tensor(labels_.at(index), torch::kInt64);

  const auto mean = torch::tensor({0.4914, 0.4822, 0.4465}).view({3,1,1});
  const auto stdd = torch::tensor({0.2470, 0.2435, 0.2616}).view({3,1,1});
  x = (x - mean) / stdd;

  return {x, y};
}
