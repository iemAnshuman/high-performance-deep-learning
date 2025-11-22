import torch
import triton
import triton.language as tl

@triton.jit
def copy_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    stride,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    src_offsets = offsets * stride
    
    mask = offsets < n_elements
    
    val = tl.load(input_ptr + src_offsets, mask=mask)
    tl.store(output_ptr + offsets, val, mask=mask)

def benchmark_memory():
    n_elements = 1024 * 1024 * 16 
    
    stride_max = 128
    src = torch.rand(n_elements * stride_max, device='cuda', dtype=torch.float32)
    dst = torch.empty(n_elements, device='cuda', dtype=torch.float32)
    
    configs = [
        triton.testing.Benchmark(
            x_names=['stride'], 
            x_vals=[1, 128], 
            line_arg='provider', 
            line_vals=['triton'], 
            line_names=['Triton'],
            ylabel='GB/s', 
            args={'n_elements': n_elements, 'BLOCK_SIZE': 1024} 
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(stride, n_elements, BLOCK_SIZE, provider):
        ms = triton.testing.do_bench(
            lambda: copy_kernel[(n_elements // BLOCK_SIZE,)](
                src, dst, n_elements, stride, BLOCK_SIZE=BLOCK_SIZE
            )
        )

        gb = n_elements * 4 / 1e9
        return gb / (ms * 1e-3)

    benchmark.run(print_data=True, show_plots=False)

if __name__ == "__main__":
    benchmark_memory()