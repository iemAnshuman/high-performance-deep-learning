import os
import time
import torch
import torch.nn as nn
import argparse
import shutil
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

# -------------------------------------------------------------------------
# Configuration & Utils
# -------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

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
# Phase 1: Real-World Training Loop
# -------------------------------------------------------------------------
def run_training_phase(model_name):
    print_header(f"Phase 1: Fine-Tuning '{model_name}'")
    device = get_device()
    
    # 1. Load Real Model
    print(f"⏳ Loading model '{model_name}'...")
    try:
        # Load config first to prevent massive downloads if just testing
        config = AutoConfig.from_pretrained(model_name)
        # Initialize empty weights to save time/bandwidth for the demo, 
        # OR remove _from_config to download full weights
        model = AutoModelForCausalLM.from_config(config).to(device)
    except Exception as e:
        print(f"❌ Failed to load '{model_name}': {e}")
        return None

    print(f"✅ Model Loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M Parameters")

    # 2. Wrap in Your Custom DDP
    setup_distributed()
    ddp_model = BucketedDDP(model, bucket_cap_mb=25)
    print("✅ Wrapped in BucketedDDP (Gradient Bucketing Enabled)")

    # 3. Real Training Step
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create dummy batch of correct size
    vocab_size = getattr(config, 'vocab_size', 32000)
    inputs = torch.randint(0, vocab_size, (1, 128)).to(device)
    
    print("🚀 Running Forward/Backward pass...")
    outputs = ddp_model(inputs)
    
    # HuggingFace models return CausalLMOutputWithPast
    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
    
    # If loss is None (e.g. no labels provided), make a dummy one
    if loss is None:
        loss = outputs.logits.mean()
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"✅ Optimizer Step Complete. Loss: {loss.item():.4f}")
    return model

# -------------------------------------------------------------------------
# Phase 2: Full Model Quantization
# -------------------------------------------------------------------------
def run_quantization_phase(model):
    print_header("Phase 2: INT4 Quantization Pipeline")
    
    if not HAS_CPP_EXT:
        print("❌ C++ Extension missing. Skipping.")
        return

    output_dir = "quantized_weights"
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print(f"📂 Saving quantized layers to: ./{output_dir}/")
    
    total_original_size = 0
    total_compressed_size = 0
    layer_count = 0
    
    start_time = time.time()
    
    # Iterate over every submodule
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 1. Grab Weights
            w = module.weight.data.detach().float()
            total_original_size += w.nbytes
            
            # 2. Quantize (Min-Max Scaling to UINT4)
            # w_q = (w - min) / (max - min) * 15
            w_min, w_max = w.min(), w.max()
            scale = (w_max - w_min) / 15.0
            zero_point = w_min
            
            w_int = ((w - zero_point) / (scale + 1e-8)).round().clamp(0, 15).to(torch.uint8)
            
            # 3. Pack with C++
            # Split into high/low nibbles for packing
            # (In a real scenario, we'd reshape. Here we just pack the first half with second half for speed)
            mid = w_int.numel() // 2
            w_flat = w_int.view(-1)
            part_a = w_flat[:mid].contiguous()
            part_b = w_flat[mid:mid*2].contiguous() # Ensure equal size
            
            if part_a.shape != part_b.shape:
                continue # Skip odd-sized layers for this demo
                
            packed = quantization_cpp.pack_int4(part_a, part_b)
            
            # 4. Save to Disk
            safe_name = name.replace(".", "_")
            fname = os.path.join(output_dir, f"{safe_name}.bin")
            with open(fname, "wb") as f:
                f.write(packed.numpy().tobytes())
                
            total_compressed_size += packed.nbytes
            layer_count += 1
            print(f"   🔹 Packed {name}: {w.nbytes/1024:.0f}KB -> {packed.nbytes/1024:.0f}KB")

    duration = time.time() - start_time
    print(f"\n✨ Quantized {layer_count} layers in {duration:.2f}s")
    print(f"📊 Original Size:   {total_original_size / 1e6:.2f} MB")
    print(f"📊 Compressed Size: {total_compressed_size / 1e6:.2f} MB")
    print(f"📉 Compression Ratio: {total_original_size / total_compressed_size:.2f}x")
    
    return output_dir

# -------------------------------------------------------------------------
# Phase 3: Zero-Copy Deployment
# -------------------------------------------------------------------------
def run_inference_phase(quantized_dir):
    print_header("Phase 3: Edge Deployment (Zero-Copy Loading)")
    
    if not quantized_dir: return

    files = [f for f in os.listdir(quantized_dir) if f.endswith('.bin')]
    if not files:
        print("❌ No quantized weights found.")
        return

    # Pick the largest layer to demonstrate
    target_file = files[0] # Just pick first one
    full_path = os.path.join(quantized_dir, target_file)
    file_size = os.path.getsize(full_path)
    
    print(f"🔋 Targeting Layer Artifact: {target_file}")
    print(f"   Physical Size on Disk: {file_size / 1024:.2f} KB")
    
    # Use your custom Loader
    loader = ZeroCopyLoader(full_path)
    
    # We load it as a flat byte array
    # In a real engine, we'd know the shape from a config file
    loaded_tensor = loader.load_tensor(offset=0, shape=(file_size,), dtype=torch.uint8)
    
    print(f"✅ Mapped to Virtual Memory at: {loaded_tensor.data_ptr()}")
    print("   (No RAM was allocated for this data yet - OS controls paging)")
    
    # Verify C++ Unpack
    if HAS_CPP_EXT:
        unpacked_a, unpacked_b = quantization_cpp.unpack_int4(loaded_tensor)
        print(f"🚀 Unpacked successfully using C++ Extension!")
        print(f"   Reconstructed Shape: {unpacked_a.shape[0] * 2} elements")

    loader.close()

# -------------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HPC-DL End-to-End Pipeline")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace Model ID (e.g. gpt2, facebook/opt-125m)")
    args = parser.parse_args()

    # 1. Prompt if not provided
    model_name = args.model
    if model_name == "gpt2": # Default
        user_input = input(f"Enter model name (Press Enter for '{model_name}'): ")
        if user_input.strip():
            model_name = user_input.strip()

    # 2. Run Pipeline
    model = run_training_phase(model_name)
    
    if model:
        q_dir = run_quantization_phase(model)
        run_inference_phase(q_dir)
        
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    print("\n✅ Workflow Complete.")