# BUILD.md

This document explains **how to build this repository**, and—more importantly—**why** it is structured the way it is. The goal is a workflow that is reproducible, reviewable, and friendly to both **local CPU development** and **cloud CUDA execution**.

---

## Build Philosophy

### 1) Source vs. Artifacts
This repo treats build outputs as **artifacts**, not source:

- **Source** (tracked in Git): code, build scripts, configuration, documentation
- **Artifacts** (NOT tracked): object files, binaries, generated build system files, caches

Artifacts are excluded via `.gitignore` because they are:
- machine-specific (paths, toolchain versions, local configuration)
- disposable (can be regenerated)
- noisy in reviews
- a common cause of “works on my machine” failures

Rule of thumb:
> If it can be produced by running a command, it does not belong in version control.

---

### 2) Out-of-Source Builds
All builds are performed in a separate directory (typically `build/`). This keeps the repository clean and allows multiple builds side-by-side (e.g., Debug vs Release).

Example:
```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Why this matters:
- easy cleanup (`rm -rf build/`)
- supports multiple configurations without conflicts
- reduces accidental commits of generated files

---

### 3) Explicit Dependency Wiring (LibTorch)
This project uses **LibTorch** (PyTorch C++ distribution), which is **not installed system-wide by default**. We intentionally keep it as an explicit, user-provided dependency.

CMake finds LibTorch via:
- `-DCMAKE_PREFIX_PATH=/path/to/libtorch`

This makes the build:
- portable across machines
- compatible with both CPU and CUDA LibTorch variants
- explicit (reviewers can see what you linked against)

---

### 4) CPU-First Development, GPU as Deployment
Local development is expected to run on CPU (e.g., laptop). GPU acceleration is treated as a deployment concern.

Key idea:
- The **same codebase** runs on CPU or CUDA.
- CUDA support depends on **which LibTorch distribution you link** and the runtime GPU environment.

This mirrors real ML infrastructure workflows:
1. develop/debug on CPU (fast iteration and low cost)
2. validate correctness and artifact outputs
3. run on GPU in cloud for performance scaling

---

## Requirements

### Build Tools
- C++17 compiler (GCC ≥ 9 or Clang ≥ 10)
- CMake ≥ 3.18
- Ninja (recommended, faster incremental builds)

On Debian/Ubuntu/Mint:
```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build git curl unzip
```

---

## LibTorch Setup

### CPU (Local Development)
Download the LibTorch **CPU** distribution and extract it, for example:
- `~/libs/libtorch`

You should have:
- `include/`
- `lib/`
- `share/cmake/Torch/`

### CUDA (Cloud GPU)
On a GPU machine, download the LibTorch **CUDA** distribution matching the machine’s CUDA runtime/driver expectations and extract it, for example:
- `/opt/libtorch`

---

## Building (CPU)

From repo root:
```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/libs/libtorch"

cmake --build build
```

Run:
```bash
./build/torchtraincpp --epochs 2 --batch 128 --data data --out runs/cpu_smoke
```

---

## Building (CUDA on Cloud)

On a CUDA machine:
```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/opt/libtorch"

cmake --build build
```

Run with CUDA enabled:
```bash
./build/torchtraincpp --cuda --epochs 10 --batch 256 --data data --out runs/gpu_run
```

---

## Runtime Linking Notes (Linux)

LibTorch is shipped as shared libraries (`.so`). If your loader cannot find them at runtime, you may see errors like:
- `libtorch_cpu.so: cannot open shared object file`

You have three common options:

### Option A: Temporary environment variable (simple)
```bash
export LD_LIBRARY_PATH="$HOME/libs/libtorch/lib:$LD_LIBRARY_PATH"
./build/torchtraincpp --epochs 1 --data data --out runs/tmp
```

### Option B: Shell profile (persistent for your user)
Append to `~/.bashrc`:
```bash
export LD_LIBRARY_PATH="$HOME/libs/libtorch/lib:$LD_LIBRARY_PATH"
```

### Option C: Use RPATH (more production-like)
If desired, the project can be configured to embed an RPATH in the binary so it can locate LibTorch without environment variables. This is a deliberate choice and can be added when preparing deployment packaging.

---

## Build Types and Debugging

### Release (default for runs)
- faster training
- representative performance measurements

### Debug (for stepping through code)
```bash
cmake -S . -B build-debug -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_PREFIX_PATH="$HOME/libs/libtorch"
cmake --build build-debug
```

---

## Clean Rebuild

Because builds are out-of-source, the cleanest rebuild is:
```bash
rm -rf build/
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$HOME/libs/libtorch"
cmake --build build
```

---

## Common Failure Modes

### 1) `Could NOT find Torch`
Cause: `CMAKE_PREFIX_PATH` not pointing at the extracted LibTorch directory.

Fix:
```bash
-DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch
```

### 2) ABI / toolchain mismatch
Symptoms:
- link errors involving `std::` symbols
- runtime crashes on startup

Fix:
- use the correct LibTorch variant for your platform/toolchain (commonly the cxx11 ABI build)
- ensure your compiler is reasonably modern

### 3) Runtime `.so` not found
Cause: loader cannot locate LibTorch shared libraries.

Fix:
- set `LD_LIBRARY_PATH` (see above), or embed RPATH

This project embeds RPATH to locate LibTorch at runtime, avoiding reliance on LD_LIBRARY_PATH.
---

## Reproducibility Checklist

A “good” build/run environment should be able to answer:
- which LibTorch distribution was used (CPU vs CUDA, version)
- which compiler version built the binary
- which command-line flags were used for the run
- where outputs were written (metrics, checkpoints)

This repo’s defaults support that by design.

---

## Why We Ignore `build/` in Git

Generated build files change constantly and frequently include absolute paths. Committing them:
- breaks other developers
- creates meaningless diffs
- makes CI unreliable
- pollutes review

The correct pattern is:
- commit build instructions (this doc + README)
- never commit build outputs
