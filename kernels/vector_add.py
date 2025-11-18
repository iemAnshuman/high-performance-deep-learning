import torch
import triton 
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def vector_add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()

    BLOCK_SIZE = 1024

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output

if __name__ == "__main__":
    size = 98432 
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    
    output_triton = vector_add(x, y)
    output_torch = x + y
    
    if torch.allclose(output_triton, output_torch):
        print("Success! Triton Vector Add matches PyTorch.")
    else:
        print("Failure! Results do not match.")
