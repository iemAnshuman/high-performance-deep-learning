import torch
import triton
import triton.language as tl
import math

@triton.jit
def gelu_approx(x):
    """
    GeLU Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Source: Hendrycks & Gimpel (2016)
    
    Constants:
        k0 = sqrt(2/pi) = 0.7978845608
        k1 = 0.044715
    """
    return 0.5 * x * (1.0 + tl.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

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
    
    Operation:
        y = GeLU(x + b)
    
    Memory optimization:
        Performs the add and activation in registers (SRAM) to avoid
        writing the intermediate (x+b) result back to HBM.
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
    Applies Fused MLP (BiasAdd + GeLU) using Triton.
    """
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_mlp_kernel[grid](x, bias, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output

if __name__ == "__main__":
    # Correctness Check
    x = torch.randn(4096 * 4096, device='cuda')
    bias = torch.randn(4096 * 4096, device='cuda')
    
    triton_out = fused_mlp(x, bias)
    
    # Reference implementation
    torch_out = torch.nn.functional.gelu(x + bias, approximate='tanh')
    
    # CERN Verification Standard: Explicit max difference check
    max_diff = torch.max(torch.abs(triton_out - torch_out))
    print(f"Max Difference: {max_diff:.6f}")
    
    if torch.allclose(triton_out, torch_out, atol=1e-3):
        print("Success! Fused MLP matches PyTorch approximation.")
    else:
        print("Mismatch! Precision error too high.")