import torch
import triton
import triton.language as tl
import math

@triton.jit
def gelu_approx(x):
    """
    GeLU Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    return 0.5 * x * (1.0 + tl.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

# --- The Autotuner ---
# Triton will benchmark these configurations and pick the best one.
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
    
    # Load input and bias
    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    
    # Fused Compute in Registers
    accum = x + bias
    output = gelu_approx(accum)
    
    # Store result
    tl.store(out_ptr + offsets, output, mask=mask)

def fused_mlp(x: torch.Tensor, bias: torch.Tensor):
    """
    Applies Fused MLP (BiasAdd + GeLU) using Triton with Autotuning.
    """
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # We no longer hardcode BLOCK_SIZE here. 
    # The autotuner chooses it and passes it as a keyword argument to the kernel.
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
        
        # This first run triggers the autotuner (benchmarking overhead happens once)
        triton_out = fused_mlp(x, bias)
        
        # Reference implementation
        torch_out = torch.nn.functional.gelu(x + bias, approximate='tanh')
        
        max_diff = torch.max(torch.abs(triton_out - torch_out))
        print(f"Max Difference: {max_diff:.6f}")
        
        if torch.allclose(triton_out, torch_out, atol=1e-3):
            print("Success! Fused MLP matches PyTorch approximation.")
        else:
            print("Mismatch! Precision error too high.")