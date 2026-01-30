import torch
import triton
import triton.language as tl

# --- The Autotuner ---
# Benchmarks different block sizes to find the fastest config for your specific GPU (T4 or H100)
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
    Optimized for high-bandwidth GPU throughput.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 1. Load input and bias (Memory Access)
    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    
    # 2. Fused Compute (SRAM/Registers)
    accum = x + bias
    
    # Inline GeLU Approximation (Manual Tanh for Robustness)
    # GeLU(x) approx = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    accum_cubed = accum * accum * accum
    inner = 0.7978845608 * (accum + 0.044715 * accum_cubed)
    
    # Manual Tanh: (e^2x - 1) / (e^2x + 1)
    # This works on ALL Triton versions
    exp_2x = tl.exp(2.0 * inner)
    tanh_val = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    output = 0.5 * accum * (1.0 + tanh_val)
    
    # 3. Store result
    tl.store(out_ptr + offsets, output, mask=mask)

# --- Autograd Wrapper (Connects Triton to PyTorch) ---
class FusedMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias):
        ctx.save_for_backward(x, bias)
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        fused_mlp_kernel[grid](x, bias, output, n_elements)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Fallback to PyTorch for backward pass simplicity
        x, bias = ctx.saved_tensors
        with torch.enable_grad():
            x_temp = x.detach().requires_grad_(True)
            bias_temp = bias.detach().requires_grad_(True)
            out_temp = torch.nn.functional.gelu(x_temp + bias_temp, approximate='tanh')
            out_temp.backward(grad_output)
        return x_temp.grad, bias_temp.grad

def fused_mlp(x: torch.Tensor, bias: torch.Tensor):
    """
    Public API: Applies Fused MLP (BiasAdd + GeLU) using Triton.
    """
    return FusedMLPFunction.apply(x, bias)