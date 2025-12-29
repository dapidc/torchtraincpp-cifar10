    #include "util/csv_logger.h"
#include <stdexcept>

CsvLogger::CsvLogger(const std::string& path) : f_(path, std::ios::out) {
  if (!f_) throw std::runtime_error("Failed to open CSV: " + path);
}

void CsvLogger::write_header() {
  f_ << "epoch,train_loss,train_acc,val_loss,val_acc\n";
  f_.flush();
}

void CsvLogger::log(int epoch, double train_loss, double train_acc, double val_loss, double val_acc) {
  f_ << epoch << "," << train_loss << "," << train_acc << "," << val_loss << "," << val_acc << "\n";
  f_.flush();
}
