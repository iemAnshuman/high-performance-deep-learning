import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, 
    input_ptr, 
    input_row_stride, 
    output_row_stride, 
    n_cols, 
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def fused_softmax(x: torch.Tensor):
    n_rows, n_cols = x.shape
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    num_warps = 4
    if BLOCK_SIZE >= 2048: num_warps = 8
    if BLOCK_SIZE >= 4096: num_warps = 16

    output = torch.empty_like(x)
    
    grid = (n_rows, )
    
    softmax_kernel[grid](
        output, x, 
        x.stride(0), output.stride(0), 
        n_cols, 
        num_warps=num_warps, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

if __name__ == "__main__":
    torch.manual_seed(0)
    
    rows, cols = 4096, 1024 # Standard BERT-like dimensions
    x = torch.randn(rows, cols, device='cuda')
    
    triton_out = fused_softmax(x)
    
    torch_out = torch.softmax(x, dim=1)
    
    if torch.allclose(triton_out, torch_out, atol=1e-2, rtol=0):
        print("Success! Fused Softmax matches PyTorch.")
    else:
        print("Mismatch!")
        print(f"Max Diff: {torch.max(torch.abs(triton_out - torch_out))}")

    def run_torch(): torch.softmax(x, dim=1)
    def run_triton(): fused_softmax(x)
    
    ms_torch = triton.testing.do_bench(run_torch)
    ms_triton = triton.testing.do_bench(run_triton)
    
    print(f"PyTorch: {ms_torch:.3f}ms")
    print(f"Triton:  {ms_triton:.3f}ms")
    print(f"Speedup: {ms_torch / ms_triton:.2f}x")