#!/usr/bin/env bash
set -euo pipefail

# Adjust this if your libtorch is elsewhere
LIBTORCH="${LIBTORCH:-$HOME/libs/libtorch}"

./scripts/get_cifar10.sh data

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$LIBTORCH"
cmake --build build

./build/torchtraincpp --epochs 10 --batch 128 --lr 0.001 --data data --out runs/cifar10_cpu
