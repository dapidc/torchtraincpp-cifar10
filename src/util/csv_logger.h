#pragma once
#include <string>
#include <fstream>

class CsvLogger {
public:
  explicit CsvLogger(const std::string& path);
  void write_header();
  void log(int epoch, double train_loss, double train_acc, double val_loss, double val_acc);

private:
  std::ofstream f_;
};
