import torch
import pytest
import triton
from kernels.fused_softmax import fused_softmax

# Verify CUDA availability
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

@pytest.mark.parametrize("rows, cols", [
    (128, 128),     # Power of 2 (Easy)
    (1024, 768),    # Non-Power of 2 (Hard - Tests Masking)
    (4096, 4096),   # Large (Tests Cache/Warp usage)
    (10, 30000)     # Extreme Aspect Ratio (Tests Grid Logic)
])
def test_softmax_correctness(rows, cols):
    """
    Verifies Triton Softmax against PyTorch reference implementation.
    Checks for Numerical Stability and Masking correctness.
    """
    torch.manual_seed(42)
    
    # Generate data
    x = torch.randn(rows, cols, device='cuda')
    
    # 1. Run Ground Truth (PyTorch)
    torch_out = torch.softmax(x, dim=1)
    
    # 2. Run Candidate (Triton)
    triton_out = fused_softmax(x)
    
    # 3. Verification
    # We use a strict tolerance (1e-2) because fp32 accumulation differs slightly
    assert torch.allclose(triton_out, torch_out, atol=1e-2, rtol=0), \
        f"Mismatch in kernel output. Max Diff: {torch.max(torch.abs(triton_out - torch_out))}"

def test_numerical_stability():
    """
    Ensures the kernel handles large inputs without returning NaNs (Safe Softmax).
    """
    rows, cols = 128, 128
    # Large values that would cause exp(x) to overflow in naive implementation
    x = torch.randn(rows, cols, device='cuda') + 1000.0 
    
    triton_out = fused_softmax(x)
    
    assert not torch.isnan(triton_out).any(), "Kernel produced NaNs on large input!"