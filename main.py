import os
import time
import shutil
import argparse
import json
import struct
import warnings
import torch
import torch.nn as nn

# Suppress specific mmap warning
warnings.filterwarnings("ignore", message="The given buffer is not writable")

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

# Import Triton Kernel
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
# Phase 1: GPU Fine-Tuning
# -------------------------------------------------------------------------
def run_training_phase(model_name):
    print_header(f"Phase 1: GPU Optimization & Training '{model_name}'")
    device = get_device()
    
    print(f"⏳ Loading model structure for '{model_name}'...")
    try:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config).to(device)
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

    print(f"✅ Model Loaded on {device.upper()}")

    if HAS_TRITON and device == 'cuda':
        print("🚀 Injecting Triton Fused Kernels...")
        dummy_input = torch.randn(32, 4096, device='cuda')
        dummy_bias = torch.randn(32, 4096, device='cuda')
        start = time.time()
        _ = fused_mlp(dummy_input, dummy_bias)
        torch.cuda.synchronize()
        print(f"   Triton Fused MLP Time: {(time.time() - start)*1000:.3f} ms")

    setup_distributed()
    ddp_model = BucketedDDP(model, bucket_cap_mb=25)
    print("✅ Wrapped in BucketedDDP")

    print("🔄 Running Backward Pass...")
    model.train()
    vocab_size = getattr(config, 'vocab_size', 32000)
    inputs = torch.randint(0, vocab_size, (1, 64)).to(device)
    
    outputs = ddp_model(inputs, labels=inputs)
    
    if hasattr(outputs, 'loss') and outputs.loss is not None:
        loss = outputs.loss
    else:
        loss = outputs.logits.mean()

    loss.backward()
    print(f"✅ Gradients Synchronized. Loss: {loss.item():.4f}")
    return model

# -------------------------------------------------------------------------
# Phase 2: INT4 Quantization (With Metadata)
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
    
    for i, (name, module) in enumerate(model.named_modules()):
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = (module.__class__.__name__ == 'Conv1D')
        
        if is_linear or is_conv1d:
            w = module.weight.data.detach().float().cpu()
            original_shape = list(w.shape)
            
            # Padding for odd-sized tensors
            padding = 0
            if w.numel() % 2 != 0:
                padding = 1
                w = torch.cat([w.view(-1), torch.zeros(1)])
            
            # Quantization
            w_min, w_max = w.min(), w.max()
            scale = (w_max - w_min) / 15.0
            if scale == 0: scale = 1.0 # Handle constant weights
            
            w_int = ((w - w_min) / (scale + 1e-8)).round().clamp(0, 15).to(torch.uint8)
            
            # Packing
            mid = w_int.numel() // 2
            w_flat = w_int.view(-1)
            packed = quantization_cpp.pack_int4(
                w_flat[:mid].contiguous(), 
                w_flat[mid:mid*2].contiguous()
            )
            
            # Save Binary
            safe_name = name.replace('.', '_')
            bin_path = os.path.join(output_dir, f"{safe_name}.bin")
            with open(bin_path, "wb") as f:
                f.write(packed.numpy().tobytes())

            # [FIX] Save Metadata for reconstruction
            meta_path = os.path.join(output_dir, f"{safe_name}.json")
            metadata = {
                "shape": original_shape,
                "min": float(w_min),
                "scale": float(scale),
                "padding": padding
            }
            with open(meta_path, "w") as f:
                json.dump(metadata, f)
            
            layer_count += 1
            print(f"   🔹 Compressed {name}")
            
            if layer_count >= 5: 
                print("   (Stopping at 5 layers for demo speed)")
                break

    return output_dir

# -------------------------------------------------------------------------
# Phase 3: Zero-Copy Inference (Correct Dequantization)
# -------------------------------------------------------------------------
def run_inference_phase(quantized_dir):
    print_header("Phase 3: Zero-Copy Inference (Virtual Memory)")
    
    if not quantized_dir: return

    # Find first bin file
    files = [f for f in os.listdir(quantized_dir) if f.endswith('.bin')]
    if not files: return
    
    bin_file = files[0]
    meta_file = bin_file.replace('.bin', '.json')
    
    target_bin = os.path.join(quantized_dir, bin_file)
    target_meta = os.path.join(quantized_dir, meta_file)
    
    if not os.path.exists(target_meta):
        print("❌ Metadata missing. Cannot dequantize.")
        return

    # Load Metadata
    with open(target_meta, 'r') as f:
        meta = json.load(f)
    
    file_size = os.path.getsize(target_bin)
    loader = ZeroCopyLoader(target_bin)
    
    try:
        # 1. Map
        loaded_tensor = loader.load_tensor(offset=0, shape=(file_size,), dtype=torch.uint8)
        print(f"✅ Mapped File: {target_bin} ({file_size/1024:.1f} KB)")
        
        # 2. Unpack & Dequantize
        if HAS_CPP_EXT:
            device = get_device()
            
            # Unpack (0-15 integers)
            a, b = quantization_cpp.unpack_int4(loaded_tensor)
            q_indices = torch.cat([a, b], dim=0).to(device).float()
            
            # Remove padding if it was added
            if meta['padding'] > 0:
                q_indices = q_indices[:-meta['padding']]
            
            # [FIX] Mathematical Dequantization
            # W ≈ scale * Q + min
            w_recon = q_indices * meta['scale'] + meta['min']
            
            # [FIX] Restore Shape
            try:
                w_recon = w_recon.view(*meta['shape'])
                print(f"   Restored Matrix Shape: {list(w_recon.shape)}")
                
                # 3. Valid Matrix Multiplication
                # Create input vector matching the last dimension of weights
                input_dim = w_recon.shape[-1]
                dummy_vec = torch.randn(input_dim, 1, device=device)
                
                # Perform MatMul (Weight @ Vector)
                res = torch.matmul(w_recon, dummy_vec)
                
                print(f"🚀 Inference Computed on {device.upper()}")
                print(f"   Output Shape: {list(res.shape)} | Mean Value: {res.mean().item():.4f}")
                
            except RuntimeError as e:
                print(f"❌ Shape Mismatch during reconstruction: {e}")

    finally:
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
        try:
            user_input = input(f"Enter model name (Press Enter for '{model_name}'): ")
            if user_input.strip(): model_name = user_input.strip()
        except EOFError:
            pass

    try:
        model = run_training_phase(model_name)
        if model:
            q_dir = run_quantization_phase(model)
            run_inference_phase(q_dir)
            
    finally:
        if torch.distributed.is_initialized():
            print("🧹 Cleaning up process group...")
            torch.distributed.destroy_process_group()
    
    print("\n✅ Optimization Pipeline Complete.")