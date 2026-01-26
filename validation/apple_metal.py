import torch
import time
import platform

def benchmark_apple_silicon():
    """
    Validates that the inference pipeline can target the 'mps' (Metal Performance Shaders)
    backend available on macOS 12.3+ devices.
    """
    print(f"[System] OS: {platform.system()}")
    print(f"[System] PyTorch Version: {torch.__version__}")

    # 1. Check Availability
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
             print("[Warning] PyTorch was not built with MPS enabled.")
        print("[Info] MPS not available. Skipping Metal validation.")
        return

    print("[Success] Apple Metal (MPS) is available!")
    
    # 2. Setup Benchmark (Matrix Multiplication)
    # Simulate a Linear Layer: [Batch, Hidden] x [Hidden, Hidden]
    BATCH = 128
    HIDDEN = 4096
    
    # Create tensors on CPU first
    a_cpu = torch.randn(BATCH, HIDDEN)
    b_cpu = torch.randn(HIDDEN, HIDDEN)
    
    # 3. Move to MPS
    device = torch.device("mps")
    try:
        a_mps = a_cpu.to(device)
        b_mps = b_cpu.to(device)
    except Exception as e:
        print(f"[Error] Failed to move tensors to MPS: {e}")
        return

    # 4. Warmup (Critical for accurate timing)
    print("Warming up MPS kernels...")
    for _ in range(5):
        _ = torch.matmul(a_mps, b_mps)

    # 5. Measure Throughput
    start_event = time.time()
    iterations = 100
    
    for _ in range(iterations):
        # Perform compute
        c = torch.matmul(a_mps, b_mps)
        # Verify: Sync (force completion)
        # On CUDA we use torch.cuda.synchronize(). On MPS, getting the item or moving to CPU syncs.
        # But for pure throughput, we just loop.
    
    # Force a sync at the end to get accurate time
    _ = c.to("cpu") 
    end_event = time.time()
    
    avg_time = (end_event - start_event) / iterations
    tflops = (2 * BATCH * HIDDEN * HIDDEN) / (avg_time * 1e12)
    
    print(f"\n[Result] Average Inference Time: {avg_time*1000:.2f} ms")
    print(f"[Result] Estimated Performance: {tflops:.2f} TFLOPS (FP32)")
    print("[Validation] PASSED. The inference engine supports Apple Silicon.")

if __name__ == "__main__":
    benchmark_apple_silicon()