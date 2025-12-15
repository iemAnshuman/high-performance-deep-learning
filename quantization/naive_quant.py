import torch

class Quantizer:
    """
    Reference implementation for affine quantization. 
    Used to verify Triton kernels later.
    """
    def __init__(self, bits=8, symmetric=True):
        self.bits = bits
        self.symmetric = symmetric
        self.qmin = -2**(bits - 1) if symmetric else 0
        self.qmax = 2**(bits - 1) - 1 if symmetric else 2**bits - 1

    def calibrate(self, x):
        """
        Calculate scale and zero-point. 
        Note: We force 0.0 to be exactly representable to preserve sparsity.
        """
        if self.symmetric:
            # For symmetric, we pin zero_point to 0. 
            # This simplifies the matmul logic significantly (no cross-terms).
            max_val = torch.max(torch.abs(x))
            scale = max_val / (2**(self.bits - 1) - 1)
            zero_point = 0
        else:
            min_val, max_val = torch.min(x), torch.max(x)
            
            # Use small epsilon to avoid div-by-zero on constant tensors
            scale = (max_val - min_val) / (self.qmax - self.qmin + 1e-6)
            
            zero_point = self.qmin - min_val / scale
            zero_point = torch.round(zero_point).clamp(self.qmin, self.qmax)
        
        return scale, int(zero_point)

    def quantize(self, x, scale, zero_point):
        # Affine quantization: x_int = round(x / S + Z)
        q_x = x / scale + zero_point
        q_x = torch.round(q_x).clamp(self.qmin, self.qmax)
        return q_x.to(torch.int8)

    def dequantize(self, q_x, scale, zero_point):
        # x_float = (x_int - Z) * S
        return (q_x.float() - zero_point) * scale

if __name__ == "__main__":
    torch.manual_seed(0)
    
    # 1. Generate random data with outliers to test robustness
    x = torch.randn(1024) * 10.0
    x[0] = 50.0 # Force an outlier
    
    # 2. Test Symmetric (Intended for weights)
    quantizer = Quantizer(bits=8, symmetric=True)
    scale, zp = quantizer.calibrate(x)
    
    q_x = quantizer.quantize(x, scale, zp)
    d_x = quantizer.dequantize(q_x, scale, zp)
    
    mse = torch.mean((x - d_x)**2)
    print(f"[Symmetric] Scale: {scale:.4f}, ZP: {zp}")
    print(f"[Symmetric] MSE: {mse:.6f}")

    # 3. Test Asymmetric (Intended for activations)
    quantizer = Quantizer(bits=8, symmetric=False)
    scale, zp = quantizer.calibrate(x)
    
    q_x = quantizer.quantize(x, scale, zp)
    d_x = quantizer.dequantize(q_x, scale, zp)
    
    mse = torch.mean((x - d_x)**2)
    print(f"[Asymmetric] Scale: {scale:.4f}, ZP: {zp}")
    print(f"[Asymmetric] MSE: {mse:.6f}")
    
    # Sanity check: Symmetric usually has higher error on asymmetric distributions (like ReLU output), 
    # but here data is gaussian centered at 0, so they should be close.
