# High-Performance Deep Learning Kernels (HPC-DL)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-green)
![Triton](https://img.shields.io/badge/triton-2.1.0-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.0-green)

A high-performance library implementing custom **OpenAI Triton kernels**, **Low-Precision Quantization schemes**, and **Distributed Communication primitives**. 

This project benchmarks the theoretical limits of consumer GPUs (e.g., RTX 4060) vs Datacenter behavior, specifically analyzing **Memory Bandwidth bottlenecks (HBM)** and **Warp Divergence** in deep learning workloads.

## Research Areas

### 1. Kernel Fusion & Optimization
Implementation of custom fused kernels to minimize HBM round-trips and maximize Arithmetic Intensity.
- **Fused Softmax**: Optimized block reduction using online normalization (Safe Softmax) with efficient register spilling management.
- **Fused MLP**: GeLU approximation fused with bias addition to keep intermediate activations in SRAM.
- **Performance**: Achieved **3.53x speedup** over PyTorch eager execution (see `kernels/assets/profiling_summary.txt`).

### 2. Low-Precision Quantization
Analysis of information loss in post-training quantization for Large Language Models.
- **Methods**: Comparative analysis of Symmetric (Weights) vs Asymmetric (Activations) Affine Quantization.
- **Format**: Simulated INT8 and INT4 bit-packing logic for custom inference engines.
- **Results**: Llama-2-7B memory footprint reduced by **69%** (13.5GB $\to$ 4.2GB) with negligible perplexity degradation on WikiText-2.

### 3. Distributed Systems Simulation
Analysis of inter-node communication latency using Ring All-Reduce topology simulation (C++ MPI).
- **Focus**: Modeling latency vs. bandwidth costs in large-scale gradient synchronization.

## Quick Start

### Installation
Clone the repository and install in editable mode:
```bash
git clone [https://github.com/iemAnshuman/high-performance-deep-learning.git](https://github.com/iemAnshuman/high-performance-deep-learning.git)
cd high-performance-deep-learning
pip install -e .

```

### Running Verification Tests

We use `pytest` for rigorous numerical stability checks against PyTorch reference implementations.

```bash
pytest tests/

```

### Benchmarking

Run the memory throughput benchmark to visualize the Roofline characteristics:

```bash
python kernels/memory_test.py

```

## Benchmarks (Roofline Analysis)

| Operation | Arithmetic Intensity (FLOPs/Byte) | Throughput (GB/s) | Bottleneck |
| --- | --- | --- | --- |
| **Vector Add** | 0.08 | 780.12 | **Memory Bound** |
| **MatMul** | 682.67 | - | **Compute Bound** |
| **Fused Softmax** | High | 84.3% of Peak | **SRAM/L2 Bound** |

## Acknowledgements

Reference implementations based on research by Hendrycks et al. (GeLU Approximation) and Dettmers et al. (BitsAndBytes Quantization).
