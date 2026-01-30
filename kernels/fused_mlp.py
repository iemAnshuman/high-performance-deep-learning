import torch
import triton
import triton.language as tl

# --- The Autotuner ---
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_mlp_kernel(
    x_ptr,      
    bias_ptr,   
    out_ptr,    
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused Bias-Add + GeLU Activation Kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 1. Load input and bias
    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    
    # 2. Fused Compute in Registers
    accum = x + bias
    
    # Inline GeLU Approximation
    # k = sqrt(2/pi) ~= 0.7978845608
    # c = 0.044715
    accum_cubed = accum * accum * accum
    inner = 0.7978845608 * (accum + 0.044715 * accum_cubed)
    
    # ROBUST TANH IMPLEMENTATION
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    # This works on ALL Triton versions (tl.exp is always available)
    exp_2x = tl.exp(2.0 * inner)
    tanh_val = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    output = 0.5 * accum * (1.0 + tanh_val)
    
    # 3. Store result
    tl.store(out_ptr + offsets, output, mask=mask)

def fused_mlp(x: torch.Tensor, bias: torch.Tensor):
    """
    Applies Fused MLP (BiasAdd + GeLU) using Triton with Autotuning.
    """
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_mlp_kernel[grid](x, bias, output, n_elements)
    
    return output

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Skipping: No CUDA device found.")
    else:
        # Correctness Check
        x = torch.randn(4096 * 4096, device='cuda')
        bias = torch.randn(4096 * 4096, device='cuda')
        
        # This first run triggers the autotuner
        triton_out = fused_mlp(x, bias)
        
        # Reference implementation
        torch_out = torch.nn.functional.gelu(x + bias, approximate='tanh')
        
        max_diff = torch.max(torch.abs(triton_out - torch_out))
        print(f"Max Difference: {max_diff:.6f}")
        
        if torch.allclose(triton_out, torch_out, atol=1e-3):
            print("Success! Fused MLP matches PyTorch approximation.")
        else:
            print("Mismatch! Precision error too high.")