import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

def get_vram_usage():
    """
    Returns memory usage in GB.
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024**3

def benchmark_model(model_id, load_in_4bit=False):
    print(f"\n--- Benchmarking: {model_id} (4-bit: {load_in_4bit}) ---")
    
    # Clean slate to ensure fair measurement
    gc.collect()
    torch.cuda.empty_cache()
    
    start_vram = get_vram_usage()
    print(f"Baseline VRAM: {start_vram:.2f} GB")
    
    start_time = time.time()
    
    try:
        # We assume the user has 'bitsandbytes' installed for load_in_4bit
        # Note: trust_remote_code=True is often needed for newer archs, adding it for safety
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            load_in_4bit=load_in_4bit,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load {model_id}: {e}")
        return

    load_time = time.time() - start_time
    loaded_vram = get_vram_usage()
    
    print(f"Loaded VRAM: {loaded_vram:.2f} GB")
    print(f"Memory Footprint: {loaded_vram - start_vram:.2f} GB")
    print(f"Load Time: {load_time:.2f}s")
    
    # Cleanup to avoid OOM on next run
    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # FP16 Baseline
    # Using a smaller proxy or the real Llama-2-7b-hf string if available
    benchmark_model("meta-llama/Llama-2-7b-hf", load_in_4bit=False)
    
    # INT4 Quantized
    # This requires bitsandbytes
    benchmark_model("meta-llama/Llama-2-7b-hf", load_in_4bit=True)
    
    # Note: For GPTQ specific benchmarking, we would use AutoGPTQ, 
    # but 'load_in_4bit' (NF4) is the modern huggingface standard.
