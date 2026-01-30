# High-Performance Deep Learning Kernels (HPC-DL)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
![Triton](https://img.shields.io/badge/triton-2.1.0-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green)
![Status](https://img.shields.io/badge/status-research_preview-yellow)

> **"The bottleneck is no longer compute; it's memory bandwidth."**

A high-performance library implementing custom **OpenAI Triton kernels**, **Low-Precision Quantization schemes**, and **Distributed Communication primitives**. 

This project benchmarks the theoretical limits of consumer GPUs (e.g., RTX 3090/4090) against Datacenter behavior, specifically analyzing **Memory Bandwidth bottlenecks (HBM)** and **Warp Divergence** in Llama-architecture workloads.

---

## Key Research Areas

### 1. Kernel Fusion & Optimization (Triton)
Standard PyTorch eager execution suffers from excessive HBM round-trips. I implemented custom fused kernels to keep data in SRAM/Registers.

- **Fused Softmax (Safe):** Optimized block reduction using online normalization to prevent FP16 overflow.
- **Fused MLP:** GeLU approximation fused with bias addition to minimize activation materialization.
- **Result:** Achieved **3.53x speedup** over PyTorch `torch.nn.functional.softmax` on large matrices (see `kernels/assets/profiling_summary.txt`).

### 2. Distributed Training Primitives (C++ / MPI)
Understanding the physics of multi-node communication is critical for 70B+ models.
- **Ring All-Reduce:** Implemented the classic bandwidth-optimal synchronization algorithm using raw MPI blocking calls (`MPI_Send`/`MPI_Recv`) to visualize the gradient flow.
- **Tensor Parallelism:** Custom `ColumnParallelLinear` layers to shard massive weight matrices across multiple GPUs for inference.

### 3. Edge Efficiency & Quantization
Making Large Language Models (LLMs) accessible on constrained hardware.
- **Zero-Copy Loading:** Implemented `mmap` based loading to map model weights directly from disk to virtual memory, bypassing CPU RAM copies.
- **Neuro-Shrink:** A CLI tool to calculate exact VRAM requirements (including KV Cache overhead) for arbitrary transformer architectures.

---

## Performance Analysis (Roofline Model)

I conducted a Roofline Analysis to determine if my kernels were Compute Bound or Memory Bound.

| Kernel Operation | Arithmetic Intensity | Throughput (GB/s) | Bottleneck |
| :--- | :--- | :--- | :--- |
| **Vector Add** | 0.08 FLOPs/Byte | 780.12 | **Memory (HBM)** |
| **MatMul (4096²)** | 682.67 FLOPs/Byte | - | **Compute (Tensor Core)** |
| **Fused Softmax** | High | **84.3% of Peak** | **L2 Cache / SRAM** |

*Artifact: Generated via `benchmarks/roofline.ipynb` on NVIDIA T4.*

---

## Quick Start

### Installation
This project requires a Linux environment with an NVIDIA GPU (Ampere or newer recommended for Triton).

```bash
# 1. Clone the repository
git clone [https://github.com/iemAnshuman/high-performance-deep-learning.git](https://github.com/iemAnshuman/high-performance-deep-learning.git)
cd high-performance-deep-learning

# 2. Install dependencies (Virtual Env recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

### Running the Distributed Simulation

To visualize the Ring All-Reduce gradient synchronization (requires `openmpi`):

```bash
make run_ring
# Output:
# [Process 0] Rank 0: Completed Step 1/3
# [Process 1] Rank 1: Completed Step 1/3 ...

```

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
│   ├── manual_ddp.py         # Custom DistributedDataParallel with Hooks
│   └── tensor_parallel.py    # Column/Row Linear Layers for 70B inference
├── kernels/
│   ├── fused_softmax.py      # Triton kernel for memory-efficient Softmax
│   ├── fused_mlp.py          # Triton kernel for GeLU + Add
│   └── vector_add.py         # Baseline bandwidth test
├── scientific/
│   └── triton_monte_carlo.py # Parallel RNG for Pi estimation (Physics demo)
├── tools/
│   └── neuro_shrink.py       # VRAM calculator for LLM deployment
└── docs/
    └── RESEARCH_LOG.md       # Engineering log of failed experiments & fixes

```
