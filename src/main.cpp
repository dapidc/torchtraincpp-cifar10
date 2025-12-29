#include <torch/torch.h>
#include <iostream>
#include <filesystem>

#include "data/cifar10_dataset.h"
#include "model/simple_cnn.h"
#include "train/trainer.h"
#include "util/csv_logger.h"
#include "util/checkpoint.h"

static TrainConfig parse_args(int argc, char** argv) {
  TrainConfig c;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto need = [&](const std::string& name) {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
      return std::string(argv[++i]);
    };

    if (a == "--epochs") c.epochs = std::stoi(need(a));
    else if (a == "--batch") c.batch_size = std::stoi(need(a));
    else if (a == "--lr") c.lr = std::stod(need(a));
    else if (a == "--cuda") c.use_cuda = true;
    else if (a == "--data") c.data_dir = need(a);
    else if (a == "--out") c.out_dir = need(a);
    else if (a == "--resume") c.resume_from = need(a);
    else if (a == "--log-every") c.log_every = std::stoi(need(a));
    else if (a == "--help") {
      std::cout <<
        "Usage: ./torchtraincpp [--epochs N] [--batch N] [--lr X] [--cuda]\n"
        "                    [--data DIR] [--out DIR] [--resume PATH] [--log-every N]\n";
      std::exit(0);
    }
  }
  return c;
}

int main(int argc, char** argv) {
  try {
    auto cfg = parse_args(argc, argv);

    torch::manual_seed(42);

    torch::Device device = torch::kCPU;
    if (cfg.use_cuda && torch::cuda::is_available()) device = torch::kCUDA;

    std::cout << "Device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

    std::filesystem::create_directories(cfg.out_dir);

    // Dataset & loaders
    auto train_ds = Cifar10Dataset(cfg.data_dir, true).map(torch::data::transforms::Stack<>());
    auto test_ds  = Cifar10Dataset(cfg.data_dir, false).map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader(
      std::move(train_ds),
      torch::data::DataLoaderOptions().batch_size(cfg.batch_size).workers(2).shuffle(true)
    );
    auto test_loader = torch::data::make_data_loader(
      std::move(test_ds),
      torch::data::DataLoaderOptions().batch_size(cfg.batch_size).workers(2)
    );

    // Model
    SimpleCnn model;
    model->to(device);

    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(cfg.lr));

    int start_epoch = 1;
    if (!cfg.resume_from.empty()) {
      int loaded_epoch = 0;
      if (load_checkpoint(cfg.resume_from, *model, opt, loaded_epoch)) {
        start_epoch = loaded_epoch + 1;
        std::cout << "Resumed from epoch " << loaded_epoch << "\n";
      } else {
        std::cout << "Resume file not found, starting fresh.\n";
      }
    }

    CsvLogger csv(cfg.out_dir + "/metrics.csv");
    csv.write_header();

    for (int epoch = start_epoch; epoch <= cfg.epochs; epoch++) {
      std::cout << "\nEpoch " << epoch << "/" << cfg.epochs << "\n";
      auto tr = train_one_epoch(*model, *train_loader, opt, device, cfg.log_every);
      auto va = eval_one_epoch(*model, *test_loader, device);

      std::cout << "train loss=" << tr.loss << " acc=" << tr.acc
                << " | val loss=" << va.loss << " acc=" << va.acc << "\n";

      csv.log(epoch, tr.loss, tr.acc, va.loss, va.acc);

      const std::string ckpt = cfg.out_dir + "/checkpoint_epoch_" + std::to_string(epoch) + ".pt";
      save_checkpoint(ckpt, *model, opt, epoch);
    }

    std::cout << "\nDone. Metrics: " << cfg.out_dir << "/metrics.csv\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    return 1;
  }
}
