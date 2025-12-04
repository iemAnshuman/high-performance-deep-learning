import torch
import triton
import triton.language as tl

@triton.jit
def gelu_approx(x):
    return 0.5 * x * (1.0 + tl.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

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
    output = gelu_approx(accum)
    
    tl.store(out_ptr + offsets, output, mask=mask)

def fused_mlp(x: torch.Tensor, bias: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_mlp_kernel[grid](x, bias, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output

if __name__ == "__main__":
    x = torch.randn(4096 * 4096, device='cuda')
    bias = torch.randn(4096 * 4096, device='cuda')
    
    triton_out = fused_mlp(x, bias)
    
    torch_out = torch.nn.functional.gelu(x + bias, approximate='tanh')
    
    if torch.allclose(triton_out, torch_out, atol=1e-3):
        print("Success! Fused MLP matches PyTorch.")
    else:
        print("Mismatch!")