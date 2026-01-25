import argparse
import sys

def parse_size(size_str):
    """
    Parses human-readable strings like '70B', '7b', '13M' into integers.
    """
    size_str = size_str.upper().replace("B", "").replace("M", "")
    try:
        if "B" in sys.argv or size_str.endswith("B"): # Logic for '70B'
             return float(size_str) * 1e9
        return float(size_str) * 1e9 # Default to Billions if just number provided
    except ValueError:
        return None

def estimate_vram(params, precision, context_len=4096, hidden_size=4096, layers=32):
    """
    Estimates VRAM usage = Model Weights + KV Cache + Activation Buffer.
    """
    # 1. Weights
    # FP32 = 4 bytes, FP16 = 2 bytes, INT8 = 1 byte, INT4 = 0.5 bytes
    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'int8': 1,
        'int4': 0.5
    }.get(precision.lower(), 2) # Default to fp16
    
    weight_usage = params * bytes_per_param
    
    # 2. KV Cache (Key-Value Cache for Attention)
    # Size = 2 * layers * hidden_size * context_len * bytes_per_param * batch_size(1)
    # Note: KV Cache is usually kept in FP16 even if weights are INT4
    kv_cache_usage = 2 * layers * hidden_size * context_len * 2 
    
    # 3. Activation Overhead (CUDA context, buffers)
    # Rule of thumb: Add ~1-2 GB for overhead
    overhead = 1.5 * 1024**3 
    
    total_bytes = weight_usage + kv_cache_usage + overhead
    return total_bytes, weight_usage, kv_cache_usage

def main():
    parser = argparse.ArgumentParser(
        description="Neuro-Shrink: Calculate if a generic LLM fits on your GPU."
    )
    parser.add_argument("model_size", type=str, help="Model size (e.g., '7B', '70B')")
    parser.add_argument("--precision", type=str, default="fp16", choices=['fp32', 'fp16', 'int8', 'int4'], 
                        help="Quantization level")
    parser.add_argument("--ctx", type=int, default=4096, help="Context Window length")
    
    args = parser.parse_args()
    
    # Human-friendly parsing
    params = parse_size(args.model_size)
    if params is None:
        print(f"Error: Could not parse model size '{args.model_size}'")
        sys.exit(1)

    total, weights, kv = estimate_vram(params, args.precision, context_len=args.ctx)
    
    # Convert to GB
    to_gb = lambda x: x / (1024**3)
    
    print(f"--- Neuro-Shrink Estimate ({args.model_size} @ {args.precision}) ---")
    print(f"Model Weights:  {to_gb(weights):.2f} GB")
    print(f"KV Cache ({args.ctx}): {to_gb(kv):.2f} GB")
    print(f"CUDA Overhead:  1.50 GB (Approx)")
    print(f"-----------------------------------")
    print(f"TOTAL VRAM REQUIRED: {to_gb(total):.2f} GB")
    
    # Helpful advice
    if to_gb(total) > 24:
        print("\n[!] Warning: This exceeds the memory of a consumer RTX 3090/4090 (24GB).")
        print("    Consider using Tensor Parallelism or INT4 quantization.")
    elif to_gb(total) <= 8:
        print("\n[+] Good news: This should fit on most modern laptops.")

if __name__ == "__main__":
    main()