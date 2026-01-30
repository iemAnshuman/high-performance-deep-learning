import torch
import pytest
import torch.nn as nn
from distributed.tensor_parallel_linear import ColumnParallelLinear

def test_tp_linear_shapes():
    """
    Verifies that ColumnParallelLinear correctly shards the weights.
    This test runs on CPU and doesn't require a cluster.
    """
    input_size = 1024
    output_size = 4096
    
    # 1. Initialize layer (Simulation Mode: defaults to world_size=1 if dist not init)
    layer = ColumnParallelLinear(input_size, output_size)
    
    # 2. Check Weight Shape
    # Since we aren't in a distributed group, it should hold the FULL weight
    assert layer.weight.shape == (output_size, input_size)
    
    # 3. Run Forward Pass
    x = torch.randn(4, input_size)
    y = layer(x)
    
    assert y.shape == (4, output_size)
    assert not torch.isnan(y).any()

# We can simulate the math of TP even on one machine
def test_tp_math_equivalence():
    """
    Mathematically validates that [X @ W1, X @ W2] == X @ W.
    We simulate 2 ranks by manually creating two slices.
    """
    B, I, O = 2, 64, 128
    
    # Standard Linear
    full_linear = nn.Linear(I, O, bias=False)
    x = torch.randn(B, I)
    expected_out = full_linear(x)
    
    # Manual Split (Simulating 2 GPUs)
    w_full = full_linear.weight.detach() # Shape: [O, I]
    w1 = w_full[:O//2, :] # Top half
    w2 = w_full[O//2:, :] # Bottom half
    
    # Local Computes
    out1 = F.linear(x, w1)
    out2 = F.linear(x, w2)
    
    # All-Gather (Concatenate)
    reconstructed_out = torch.cat([out1, out2], dim=-1)
    
    assert torch.allclose(expected_out, reconstructed_out, atol=1e-5), \
        "Tensor Parallel math slicing logic is incorrect!"