# torchtraincpp-cifar10

A **production-grade C++17 training pipeline** for **CIFAR-10** using **LibTorch**, designed to run **unchanged** on CPU (local development) and CUDA GPU (cloud or on-prem).

This repository emphasizes **systems-level ML engineering**: reproducibility, correctness, explicit device control, and clean build tooling — rather than research novelty.

---

## Design Goals

- **Single codebase** for CPU and GPU execution
- **Explicit device management** (no hidden magic)
- **Deterministic & reproducible** training runs
- **Readable, auditable C++** suitable for production environments
- **Minimal dependencies** (LibTorch only)

This project intentionally avoids Python and dynamic runtime dependencies.

---

## Key Capabilities

- CIFAR-10 training using official binary dataset
- Modular C++17 architecture (data / model / training / utilities)
- CPU ↔ CUDA switching via runtime flag
- Training + validation loops with accuracy metrics
- Checkpoint save & resume
- CSV-based metrics logging (tool-agnostic)
- CMake-based build (CPU and CUDA variants)

---

## Model Summary

- **Input:** 3×32×32 RGB images
- **Architecture:** Small CNN
  - 2× Convolution + ReLU + MaxPool
  - 2× Fully Connected
  - Dropout regularization
- **Loss:** Cross-entropy
- **Optimizer:** Adam

The model is intentionally simple to keep the focus on **infrastructure and training mechanics**, not SOTA performance.

---

## Repository Layout

```
torchtraincpp-cifar10/
├── CMakeLists.txt
├── README.md
├── scripts/
│   ├── get_cifar10.sh        # Dataset fetch (official binary format)
│   └── run_cpu.sh            # Local CPU build + run
├── src/
│   ├── main.cpp              # Entry point + orchestration
│   ├── data/                 # Dataset loading & preprocessing
│   ├── model/                # Neural network definitions
│   ├── train/                # Training / evaluation loops
│   └── util/                 # Logging, checkpoints, helpers
```

---

## Requirements

### Local Development (CPU)
- Linux (tested on Linux Mint)
- GCC ≥ 9 or Clang ≥ 10
- CMake ≥ 3.18
- LibTorch **CPU** distribution

### GPU Execution (Cloud / On‑Prem)
- NVIDIA GPU
- Compatible NVIDIA driver
- LibTorch **CUDA** distribution (matching CUDA version)

---

## Build & Run (CPU)

### Install toolchain
```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build git curl unzip
```

### Download LibTorch (CPU)
Download the **LibTorch CPU (Linux)** build from PyTorch and extract to:

```
~/libs/libtorch
```

### Build & train
```bash
git clone https://github.com/<your-username>/torchtraincpp-cifar10.git
cd torchtraincpp-cifar10

./scripts/run_cpu.sh
```

This performs:
1. CIFAR-10 download
2. Release-mode build
3. Training run on CPU
4. Metrics + checkpoint output

---

## Metrics & Artifacts

- **Metrics:** `runs/<run-name>/metrics.csv`
- **Checkpoints:** `checkpoint_epoch_*.pt`

CSV schema:
```
epoch,train_loss,train_acc,val_loss,val_acc
```

This format is intentionally tool-agnostic (Excel, pandas, gnuplot, etc.).

---

## Runtime Configuration

Supported CLI flags:

```
--epochs N
--batch N
--lr X
--cuda
--data DIR
--out DIR
--resume PATH
--log-every N
```

Example:
```bash
./build/torchtraincpp --epochs 50 --batch 256 --lr 0.0005
```

---

## GPU Run on Cloud (LibTorch CUDA)

### 1) Provision a GPU machine
Any CUDA-capable Linux environment works:
- Google Colab
- RunPod / Vast.ai / Paperspace
- Self-managed GPU server

Verify:
```bash
nvidia-smi
```

---

### 2) Install LibTorch (CUDA)
Download the **LibTorch CUDA** build matching the system CUDA version from PyTorch.

Extract to:
```
/opt/libtorch
```

---

### 3) Build against CUDA LibTorch
```bash
cmake -S . -B build -G Ninja   -DCMAKE_BUILD_TYPE=Release   -DCMAKE_PREFIX_PATH=/opt/libtorch

cmake --build build
```

---

### 4) Run with CUDA enabled
```bash
./build/torchtraincpp --cuda --epochs 50 --batch 256
```

Expected output:
```
Device: CUDA
```

No source code changes are required.

---

## Engineering Notes

- CPU-first development minimizes iteration cost
- GPU execution is treated as a deployment concern
- Device placement is explicit and auditable
- Checkpoints include optimizer state for full resume

This mirrors real-world ML infrastructure workflows.

---

## Non‑Goals

- Distributed training
- Automated hyperparameter search
- State-of-the-art accuracy tuning

These can be layered on top if needed.

---

## License

MIT License

---

## Author

Developed as a **C++ / GPU ML infrastructure showcase**.
