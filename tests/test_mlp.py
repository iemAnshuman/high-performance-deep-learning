import torch
import pytest
import torch.nn.functional as F

# Try to import the kernel, skip if Triton/CUDA is missing (e.g. on Mac)
try:
    from kernels.fused_mlp import fused_mlp
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

@pytest.mark.skipif(not HAS_CUDA, reason="Requires CUDA and Triton")
def test_fused_mlp_correctness():
    """
    Verifies that the Fused MLP (Bias + GeLU) matches PyTorch's implementation.
    """
    torch.manual_seed(42)
    
    # Dimensions
    BATCH = 128
    DIM = 1024
    
    # Inputs (on GPU)
    x = torch.randn(BATCH * DIM, device='cuda')
    bias = torch.randn(BATCH * DIM, device='cuda')
    
    # 1. Triton Kernel Output
    triton_out = fused_mlp(x, bias)
    
    # 2. PyTorch Reference
    # Note: We must use the 'tanh' approximation to match the kernel logic
    ref_out = F.gelu(x + bias, approximate='tanh')
    
    # 3. Verify
    # FP16 operations usually require slightly looser tolerance
    assert torch.allclose(triton_out, ref_out, atol=1e-3, rtol=1e-3), \
        "Fused MLP output does not match PyTorch reference!"

@pytest.mark.skipif(not HAS_CUDA, reason="Requires CUDA and Triton")
def test_mlp_boundary_values():
    """
    Test extreme values to ensure no NaNs or weird behavior.
    """
    x = torch.tensor([100.0, -100.0, 0.0], device='cuda')
    bias = torch.zeros_like(x)
    
    out = fused_mlp(x, bias)
    
    # GeLU(100) -> 100
    # GeLU(-100) -> 0
    # GeLU(0) -> 0
    expected = torch.tensor([100.0, 0.0, 0.0], device='cuda')
    
    assert torch.allclose(out, expected, atol=1e-1)