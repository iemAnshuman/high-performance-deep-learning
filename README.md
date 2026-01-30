# High-Performance Deep Learning Kernels (HPC-DL)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
![Triton](https://img.shields.io/badge/triton-2.1.0-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green)
![Status](https://img.shields.io/badge/status-research_preview-yellow)

> **"The bottleneck is no longer compute; it's memory bandwidth."**

A high-performance library implementing custom **OpenAI Triton kernels**, **Low-Precision Quantization schemes**, and **Distributed Communication primitives**. 

This project benchmarks the theoretical limits of hardware accelerators against Datacenter behavior, specifically analyzing **Memory Bandwidth bottlenecks (HBM)** and **Warp Divergence** in Llama-architecture workloads.

---

## Key Research Areas

### 1. Kernel Fusion & Optimization (Triton)
Standard PyTorch eager execution suffers from excessive HBM round-trips. I implemented custom fused kernels to keep data in SRAM/Registers.

- **Fused Softmax:** Optimized block reduction using online normalization to prevent FP16 overflow.
- **Fused MLP:** GeLU approximation fused with bias addition to minimize activation materialization.
- **Result:** Achieved **3.53x speedup** over PyTorch `torch.nn.functional.softmax` on large matrices (see `kernels/assets/profiling_summary.txt`).

### 2. Distributed Training Primitives (C++ / MPI)
Understanding the physics of multi-node communication is critical for 70B+ models.
- **Ring All-Reduce:** Implemented the classic bandwidth-optimal synchronization algorithm using raw MPI blocking calls (`MPI_Send`/`MPI_Recv`) to visualize gradient flow.
- **Gradient Bucketing:** Custom DDP wrapper that aggregates gradients into optimal 25MB chunks, hiding network latency during the backward pass.

### 3. Systems-Level Quantization (C++)
Bypassing the Python interpreter for high-throughput compression.
- **C++ Bit Packing:** A custom PyTorch C++ extension to pack/unpack INT4 weights instantly, reducing memory footprint by 8x compared to FP32.
- **Zero-Copy Inference:** Implemented `mmap`-based loading to map model weights directly from disk to virtual memory, enabling 70B+ model inference on consumer hardware (e.g., RTX 3090/4090) with limited VRAM.

---

## Performance Analysis (Roofline Model)

I conducted a Roofline Analysis to determine if my kernels were Compute Bound or Memory Bound.

| Kernel Operation | Arithmetic Intensity | Bottleneck |
| :--- | :--- | :--- |
| **Vector Add** | 0.08 FLOPs/Byte | **Memory (HBM)** |
| **MatMul (4096²)** | 682.67 FLOPs/Byte | **Compute (Tensor Core)** |
| **Fused Softmax** | High | **L2 Cache / SRAM** |

*Artifact: Generated via `benchmarks/roofline.ipynb`.*

---

## Quick Start

### Installation
This project requires a Linux environment with an NVIDIA GPU (Ampere or newer recommended for Triton).

```bash
# 1. Clone the repository
git clone [https://github.com/iemAnshuman/high-performance-deep-learning.git](https://github.com/iemAnshuman/high-performance-deep-learning.git)
cd high-performance-deep-learning

# 2. Compile C++ Extensions (MPI Ring & Quantization)
pip install -e . --no-build-isolation

```

### Running the End-to-End Pipeline

We provide a demo script that simulates Training (Phase 1), Compression (Phase 2), and Inference (Phase 3).

```bash
python demo_pipeline.py

```

*Note: On systems without CUDA, the pipeline automatically falls back to CPU simulation mode for logic verification.*

### Verifying Triton Kernels

Run the `pytest` suite to ensure numerical stability against PyTorch reference implementations:

```bash
pytest tests/

```

---

## 📂 Repository Structure

```text
├── distributed/
│   ├── ring_reduce.cpp       # MPI implementation of Ring All-Reduce
│   ├── manual_ddp.py         # Custom DDP with Gradient Bucketing
│   └── tensor_parallel.py    # Column/Row Linear Layers for 70B inference
├── kernels/
│   ├── fused_softmax.py      # Triton kernel for memory-efficient Softmax
│   ├── fused_mlp.py          # Triton kernel for GeLU + Add w/ Autotuning
│   └── vector_add.py         # Baseline bandwidth test
├── quantization/
│   ├── cpp_packing.cpp       # C++ Extension for INT4 Bit-Packing
│   └── naive_quant.py        # Python reference implementation
├── inference/
│   └── mmap_loader.py        # Zero-Copy Loader using OS Page Cache
└── docs/
    └── RESEARCH_LOG.md       # Engineering log of experiments & trade-offs

```

## Acknowledgements

* **OpenAI Triton Team** for the blocked algorithms tutorial.
