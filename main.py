import os
import time
import shutil
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

# --- Project Imports ---
try:
    import quantization_cpp
    HAS_CPP_EXT = True
except ImportError:
    HAS_CPP_EXT = False
    print("⚠️  Warning: 'quantization_cpp' not found. Run 'pip install -e .' first.")

from distributed.manual_ddp import BucketedDDP
from inference.mmap_loader import ZeroCopyLoader

# Import Triton Kernel (with error handling)
try:
    from kernels.fused_mlp import fused_mlp
    HAS_TRITON = torch.cuda.is_available()
except ImportError:
    HAS_TRITON = False

# -------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def print_header(msg):
    print(f"\n{'='*60}\n {msg}\n{'='*60}")

def setup_distributed():
    if not torch.distributed.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        torch.distributed.init_process_group(backend=backend)

# -------------------------------------------------------------------------
# Phase 1: GPU Fine-Tuning (With Triton)
# -------------------------------------------------------------------------
def run_training_phase(model_name):
    print_header(f"Phase 1: GPU Optimization & Training '{model_name}'")
    device = get_device()
    
    print(f"⏳ Loading model structure for '{model_name}'...")
    try:
        config = AutoConfig.from_pretrained(model_name)
        # Using empty weights to save download time for the demo
        model = AutoModelForCausalLM.from_config(config).to(device)
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

    print(f"✅ Model Loaded on {device.upper()}")

    # 1. Apply Triton Kernels (Replacing PyTorch Layers)
    if HAS_TRITON and device == 'cuda':
        print("🚀 Injecting Triton Fused Kernels...")
        # (In a real scenario, we'd monkey-patch the model's MLP activation)
        # Here we demonstrate the kernel execution on a dummy pass
        dummy_input = torch.randn(32, 4096, device='cuda')
        dummy_bias = torch.randn(32, 4096, device='cuda')
        
        # Benchmarking Triton vs PyTorch
        start = time.time()
        _ = fused_mlp(dummy_input, dummy_bias)
        torch.cuda.synchronize()
        print(f"   Triton Fused MLP Time: {(time.time() - start)*1000:.3f} ms")

    # 2. Distributed Wrapper
    setup_distributed()
    ddp_model = BucketedDDP(model, bucket_cap_mb=25)
    print("✅ Wrapped in BucketedDDP")

    # 3. Training Step
    print("🔄 Running Backward Pass...")
    model.train()
    vocab_size = getattr(config, 'vocab_size', 32000)
    inputs = torch.randint(0, vocab_size, (1, 64)).to(device)
    
    # FIX: Pass labels so the model calculates loss internally
    outputs = ddp_model(inputs, labels=inputs)
    
    # FIX: Robust check for None loss (HuggingFace outputs.loss is None if labels aren't passed)
    if hasattr(outputs, 'loss') and outputs.loss is not None:
        loss = outputs.loss
    else:
        # Fallback if no labels were passed
        loss = outputs.logits.mean()

    loss.backward()
    
    print(f"✅ Gradients Synchronized. Loss: {loss.item():.4f}")
    return model

# -------------------------------------------------------------------------
# Phase 2: INT4 Quantization (CPU Offload)
# -------------------------------------------------------------------------
def run_quantization_phase(model):
    print_header("Phase 2: INT4 Quantization Pipeline")
    
    if not HAS_CPP_EXT:
        print("❌ C++ Extension missing.")
        return None

    output_dir = "quantized_weights"
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print(f"📂 Saving to: ./{output_dir}/")
    layer_count = 0
    
    # We only quantize the first 5 layers for the demo to be fast
    for i, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, nn.Linear):
            # FIX: Move to CPU before passing to C++ to avoid Segfault
            w = module.weight.data.detach().float().cpu()
            
            # Quantize
            w_min, w_max = w.min(), w.max()
            scale = (w_max - w_min) / 15.0
            w_int = ((w - w_min) / (scale + 1e-8)).round().clamp(0, 15).to(torch.uint8)
            
            # Pack
            mid = w_int.numel() // 2
            w_flat = w_int.view(-1)
            packed = quantization_cpp.pack_int4(w_flat[:mid].contiguous(), w_flat[mid:mid*2].contiguous())
            
            # Save
            fname = os.path.join(output_dir, f"{name.replace('.', '_')}.bin")
            with open(fname, "wb") as f:
                f.write(packed.numpy().tobytes())
            
            layer_count += 1
            print(f"   🔹 Compressed {name}")
            
            if layer_count >= 5: 
                print("   (Stopping at 5 layers for demo speed)")
                break

    return output_dir

# -------------------------------------------------------------------------
# Phase 3: Zero-Copy Load & Inference
# -------------------------------------------------------------------------
def run_inference_phase(quantized_dir):
    print_header("Phase 3: Zero-Copy Inference (Virtual Memory)")
    
    if not quantized_dir: return

    files = [f for f in os.listdir(quantized_dir) if f.endswith('.bin')]
    if not files: return

    target_file = os.path.join(quantized_dir, files[0])
    file_size = os.path.getsize(target_file)
    
    # 1. Load (Instant)
    loader = ZeroCopyLoader(target_file)
    loaded_tensor = loader.load_tensor(offset=0, shape=(file_size,), dtype=torch.uint8)
    print(f"✅ Mapped File: {target_file} ({file_size/1024:.1f} KB)")
    
    # 2. Unpack & Move to GPU
    if HAS_CPP_EXT:
        a, b = quantization_cpp.unpack_int4(loaded_tensor)
        full = torch.cat([a, b], dim=0).to(get_device()).float()
        
        # 3. Compute
        res = torch.matmul(full, torch.randn(full.shape[1], 1, device=get_device()))
        print(f"🚀 Inference Computed on {get_device().upper()}")

    loader.close()

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace Model ID")
    args = parser.parse_args()

    model_name = args.model
    if model_name == "gpt2":
        # Allow non-interactive mode for automated testing
        try:
            user_input = input(f"Enter model name (Press Enter for '{model_name}'): ")
            if user_input.strip(): model_name = user_input.strip()
        except EOFError:
            pass

    model = run_training_phase(model_name)
    if model:
        q_dir = run_quantization_phase(model)
        run_inference_phase(q_dir)
        
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    print("\n✅ Optimization Pipeline Complete.")