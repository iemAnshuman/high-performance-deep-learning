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
    """
    Fused Softmax Kernel with Online Normalization (Safe Softmax).
    
    Mathematical Formulation:
    The standard softmax is unstable for large inputs. We use the "Safe Softmax":
        $$ y_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}} $$
    where \( m = \max(x) \). This ensures the exponent is always \(\le 0\), preventing overflow.

    Memory Optimization Strategy:
    1. Row-Major Load: We load an entire row into SRAM to maximize memory coalescing.
    2. Register Reduction: Max and Sum reductions happen in registers (SRAM), avoiding 
       expensive round-trips to HBM (High Bandwidth Memory).
    
    Complexity:
        Time: O(N)
        Space: O(1) (In-place if needed, or minimal overhead)
    """
    # 1. Setup row pointers
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # 2. Handle memory offsets and masking for non-power-of-2 columns
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    
    # 3. Load data into SRAM (Registers)
    # Masked load ensures we don't read out of bounds. -inf padding for max logic.
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # 4. Compute Max for Numerical Stability (Safe Softmax)
    row_minus_max = row - tl.max(row, axis=0)
    
    # 5. Compute Exponentials
    numerator = tl.exp(row_minus_max)
    
    # 6. Compute Sum (Denominator)
    denominator = tl.sum(numerator, axis=0)
    
    # 7. Final Division & Store
    softmax_output = numerator / denominator
    
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def fused_softmax(x: torch.Tensor):
    """
    Frontend wrapper for the Triton Softmax Kernel.
    Auto-tunes BLOCK_SIZE based on column width to optimize occupancy.
    """
    n_rows, n_cols = x.shape
    
    # Block size must be a power of 2 greater than n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Heuristic for Warp management based on Block Size
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