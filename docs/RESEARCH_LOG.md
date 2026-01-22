# Research Log: Triton-1.58 & Distributed Systems

## 2025-12-08: The Bottleneck
Profiling the baseline distillation loop using Nsight Compute. 
Results are bad.
- The standard `KLDivLoss` in PyTorch materializes the full `(Batch, Vocab)` probability matrix.
- For Llama-70B vocab (128k), this is huge. 
- Memory Bandwidth utilization is pegged at 98%, but Tensor Core utilization is < 20%.
- **Hypothesis:** Fusing the Softmax + Log + KL computation into a single kernel will keep the massive probability matrix in SRAM/Registers and avoid HBM trips.

## 2026-01-05: Distributed Ring Logic
Started implementing the Ring All-Reduce in C++ (MPI).
- **Issue:** Deadlocks when running on 4 processes. 
- **Fix:** Realized that if everyone calls `MPI_Send` at once, there's no buffer. Switched to the "Even/Odd" strategy (Even ranks send, Odd ranks recv). 
- **Note:** TCP latency on the university cluster is high (~50us). RDMA (InfiniBand) would be better, but sticking to TCP for compatibility.

## 2026-01-15: Quantization Experiments
Tried to implement the 1.58-bit quantization (ternary weights: -1, 0, 1).
- **Failure:** The accuracy drops off a cliff for the 7B model. 
- The paper suggests "Rotary Embeddings" might be sensitive to quantization noise.
- **Decision:** Focusing on the *System Architecture* (Inference Engine) rather than model quality for now. The goal is to build the *runner*, not train the model from scratch.

## 2026-01-18: Triton Kernel Success
The fused kernel works. 
- Reduced VRAM usage for the distillation step by ~40%.
- Speedup is roughly 3x over the PyTorch baseline.
- **Gotcha:** `tl.atomic_add` is necessary when reducing across blocks, but it's slow. Optimized by doing a block-level reduction first, then writing to a small temporary buffer.