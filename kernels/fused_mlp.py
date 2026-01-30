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
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    
    accum = x + bias
    
    # Inline GeLU Approx (tanh replacement for stability)
    accum_cubed = accum * accum * accum
    inner = 0.7978845608 * (accum + 0.044715 * accum_cubed)
    exp_2x = tl.exp(2.0 * inner)
    tanh_val = (exp_2x - 1.0) / (exp_2x + 1.0)
    output = 0.5 * accum * (1.0 + tanh_val)
    
    tl.store(out_ptr + offsets, output, mask=mask)

# --- Autograd Wrapper ---
class FusedMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias):
        # Save for backward
        ctx.save_for_backward(x, bias)
        
        # 1. Run Fast Triton Kernel
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        fused_mlp_kernel[grid](x, bias, output, n_elements)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, bias = ctx.saved_tensors
        # 2. Use PyTorch for Backward (easier than writing Triton backward)
        # We recompute the local graph to let PyTorch handle the derivative math
        with torch.enable_grad():
            x_temp = x.detach().requires_grad_(True)
            bias_temp = bias.detach().requires_grad_(True)
            # Match the approximation used in the kernel
            out_temp = torch.nn.functional.gelu(x_temp + bias_temp, approximate='tanh')
            out_temp.backward(grad_output)
            
        return x_temp.grad, bias_temp.grad

# Public API
def fused_mlp(x: torch.Tensor, bias: torch.Tensor):
    return FusedMLPFunction.apply(x, bias)