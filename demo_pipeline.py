import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# --- Project Imports ---
try:
    import quantization_cpp
    HAS_CPP_EXT = True
except ImportError:
    HAS_CPP_EXT = False
    print("⚠️  Warning: 'quantization_cpp' not found. Run 'pip install -e .' first.")

from distributed.manual_ddp import BucketedDDP
from inference.mmap_loader import ZeroCopyLoader

# --- Triton Kernels ---
try:
    # We now import the Robust wrappers that support Autograd
    from distillation.fused_kl_div import fused_kl_loss
    from kernels.fused_mlp import fused_mlp
    HAS_TRITON = torch.cuda.is_available()
except ImportError as e:
    print(f"⚠️  Triton Import Error: {e}")
    HAS_TRITON = False

# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def setup_distributed():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend)

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_device():
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

# -------------------------------------------------------------------------
# Phase 1: Pipeline Optimization
# -------------------------------------------------------------------------
def phase_1_training_simulation():
    device = get_device()
    print(f"\n[🏗️  Phase 1] Training & Optimization Pipeline ({device.upper()})")
    
    setup_distributed()
    
    # 1. Models
    teacher = nn.Linear(4096, 4096).to(device)
    student = nn.Linear(4096, 4096).to(device)
    
    # 2. Distributed Wrapper
    ddp_student = BucketedDDP(student, bucket_cap_mb=1) 
    print("   ✅ Student Model wrapped in BucketedDDP")

    # 3. Batch
    inputs = torch.randn(32, 4096).to(device)
    
    # 4. Forward Pass
    with torch.no_grad():
        teacher_logits = teacher(inputs)
    
    # [OPTIMIZATION] Fused MLP
    if HAS_TRITON and device == 'cuda':
        linear_out = ddp_student(inputs)
        bias = torch.randn_like(linear_out)
        # Using the Autograd-Compatible Wrapper
        student_logits = fused_mlp(linear_out, bias)
        print("   🚀 Executed Fused MLP (Triton) for activation")
    else:
        student_logits = ddp_student(inputs)
    
    # 5. Distillation Loss
    if HAS_TRITON and device == 'cuda':
        print("   🚀 Executed Fused KL Divergence (Triton) for loss")
        loss = fused_kl_loss(teacher_logits, student_logits)
    else:
        print("   ⚠️  Running in CPU Simulation Mode")
        T_log_probs = F.log_softmax(teacher_logits, dim=-1)
        S_log_probs = F.log_softmax(student_logits, dim=-1)
        loss = F.kl_div(S_log_probs, T_log_probs, reduction='batchmean', log_target=True)

    # 6. Backward (Now works because we added Autograd!)
    loss.backward()
    print(f"   ✅ Backward Pass Complete. Loss: {loss.item():.4f}")
    
    return student.weight.detach().cpu()

# -------------------------------------------------------------------------
# Phase 2: Quantization
# -------------------------------------------------------------------------
def phase_2_compression(weight_tensor):
    print("\n[📉 Phase 2] Quantization & Compression")
    
    if not HAS_CPP_EXT:
        print("   ❌ Skipping C++ packing (Extension missing).")
        return None, None

    print(f"   Original Size: {weight_tensor.nbytes / 1024 / 1024:.2f} MB")
    
    q_data = (weight_tensor - weight_tensor.min()) 
    q_data = (q_data / q_data.max() * 15).to(torch.uint8)
    
    mid = q_data.shape[0] // 2
    part_a = q_data[:mid].contiguous()
    part_b = q_data[mid:].contiguous()
    
    start = time.time()
    packed_tensor = quantization_cpp.pack_int4(part_a, part_b)
    end = time.time()
    
    print(f"   ⚡ C++ Packed 4-bit Tensor in {(end - start)*1000:.2f} ms")
    print(f"   Packed Size:   {packed_tensor.nbytes / 1024 / 1024:.2f} MB")
    
    filename = "student_model_int4.bin"
    with open(filename, "wb") as f:
        f.write(packed_tensor.numpy().tobytes())
    
    return filename, packed_tensor.shape

# -------------------------------------------------------------------------
# Phase 3: Inference
# -------------------------------------------------------------------------
def phase_3_inference(filename, shape):
    print("\n[📱 Phase 3] Optimized Inference Runtime")
    
    if not filename: return

    loader = ZeroCopyLoader(filename)
    packed_weights = loader.load_tensor(offset=0, shape=shape, dtype=torch.uint8)
    print("   ✅ Zero-Copy Load (Virtual Memory Mapped)")

    if HAS_CPP_EXT:
        unpacked_a, unpacked_b = quantization_cpp.unpack_int4(packed_weights)
        full_weight = torch.cat([unpacked_a, unpacked_b], dim=0)
        
        device = get_device()
        edge_weight = full_weight.to(device).float()
        
        dummy_input = torch.randn(32, edge_weight.shape[1]).to(device)
        output = F.linear(dummy_input, edge_weight)
        
        print(f"   🚀 Inference successful on {device.upper()}")
    
    loader.close()
    if os.path.exists(filename):
        os.remove(filename)

if __name__ == "__main__":
    print("===============================================================")
    print("   HPC-DL: Optimized Training & Inference Pipeline")
    print("===============================================================")
    
    if dist.is_initialized(): dist.destroy_process_group()

    weights = phase_1_training_simulation()
    model_file, shape = phase_2_compression(weights)
    phase_3_inference(model_file, shape)
    
    cleanup_distributed()