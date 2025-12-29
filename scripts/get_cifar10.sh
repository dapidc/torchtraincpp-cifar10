#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ -d "cifar-10-batches-bin" ]; then
  echo "CIFAR-10 already present: $DATA_DIR/cifar-10-batches-bin"
  exit 0
fi

echo "Downloading CIFAR-10 (binary)..."
curl -L -o cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzf cifar-10-binary.tar.gz
rm cifar-10-binary.tar.gz

echo "Done: $DATA_DIR/cifar-10-batches-bin"
