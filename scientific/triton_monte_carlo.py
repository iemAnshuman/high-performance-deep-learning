import torch
import triton
import triton.language as tl
import math

@triton.jit
def monte_carlo_pi_kernel(
    output_ptr,
    n_samples,
    seed,
    BLOCK_SIZE: tl.constexpr
):
    """
    Estimates Pi by generating random points in a unit square.
    Each block processes BLOCK_SIZE samples.
    """
    # 1. unique identifier for this block of threads
    pid = tl.program_id(axis=0)
    
    # 2. Generate random offsets
    # We use the block ID and seed to ensure every block gets different numbers.
    # Note: Triton's rand is a bit tricky. We hash the ID to get a seed.
    offs = tl.arange(0, BLOCK_SIZE)
    
    # Generate X and Y coordinates in [0, 1)
    # scaling seed by a large prime to decorrelate blocks
    rng_seed = seed + pid * 999999 
    x = tl.rand(rng_seed, offs)
    y = tl.rand(rng_seed + 1, offs) # offset seed for Y so it's different from X

    # 3. Check Condition: x^2 + y^2 <= 1
    dist_sq = x*x + y*y
    inside_circle = dist_sq <= 1.0

    # 4. Sum locally
    # We sum up how many points in this block fell inside
    count = tl.sum(inside_circle, axis=0)

    # 5. Atomic Add to global memory
    # Since multiple blocks write to the same output pointer, we need atomic_add
    tl.atomic_add(output_ptr, count)

def estimate_pi(n_samples=10_000_000):
    # Setup
    output = torch.zeros(1, dtype=torch.int32, device='cuda')
    grid = lambda meta: (triton.cdiv(n_samples, meta['BLOCK_SIZE']),)
    
    # Launch
    # We treat n_samples as the number of threads essentially
    monte_carlo_pi_kernel[grid](output, n_samples, seed=12345, BLOCK_SIZE=1024)
    
    total_inside = output.item()
    pi_estimate = 4.0 * (total_inside / n_samples)
    return pi_estimate

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Skipping: No GPU available for Triton.")
        exit(0)

    N = 10_000_000 # 10 Million points
    print(f"Running Monte Carlo Simulation with {N} samples...")
    
    pi = estimate_pi(N)
    
    error = abs(pi - math.pi)
    print(f"Estimated Pi: {pi:.6f}")
    print(f"Actual Pi:    {math.pi:.6f}")
    print(f"Error:        {error:.6f}")
    
    if error < 0.001:
        print("SUCCESS: Converged within tolerance.")
    else:
        print("WARNING: High error. Try increasing n_samples.")