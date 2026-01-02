import torch

class Quantizer:
    """
    Affine Quantization Reference Implementation.
    
    Theory:
        Quantization maps a floating point value \( x \in [x_{min}, x_{max}] \) to 
        an integer \( q \in [0, 2^b-1] \) (asymmetric) or \([-2^{b-1}, 2^{b-1}-1]\) (symmetric).
        
        Formula:
            $$ q = \text{clamp}(\text{round}(x / S + Z), q_{min}, q_{max}) $$
            $$ x_{dequant} = (q - Z) * S $$
        
    Args:
        bits (int): Bit-width (e.g., 8, 4).
        symmetric (bool): If True, maps 0.0 float to 0 int (Scale-only).
                          If False, uses Zero-Point (Scale + Shift).
    """
    def __init__(self, bits=8, symmetric=True):
        self.bits = bits
        self.symmetric = symmetric
        if symmetric:
            self.qmin = -2**(bits - 1)
            self.qmax = 2**(bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2**bits - 1

    def calibrate(self, x):
        """
        Computes quantization parameters (Scale and Zero-Point).
        """
        if self.symmetric:
            # Symmetric: Centered on 0. Used for Weights.
            # Range: [-max(|x|), max(|x|)]
            max_val = torch.max(torch.abs(x))
            # Protect against divide-by-zero
            scale = max_val / (2**(self.bits - 1) - 1 + 1e-8)
            zero_point = 0
        else:
            # Asymmetric: Uses min/max. Used for Activations (e.g. ReLU output).
            min_val, max_val = torch.min(x), torch.max(x)
            scale = (max_val - min_val) / (self.qmax - self.qmin + 1e-8)
            zero_point = self.qmin - min_val / scale
            zero_point = torch.round(zero_point).clamp(self.qmin, self.qmax)
        
        return scale, int(zero_point)

    def quantize(self, x, scale, zero_point):
        """
        Projects float32 tensor to quantized integer domain.
        """
        q_x = x / scale + zero_point
        q_x = torch.round(q_x).clamp(self.qmin, self.qmax)
        
        # Simulating INT8 storage using float container for PyTorch operations
        return q_x

    def dequantize(self, q_x, scale, zero_point):
        """
        Recovers approximation of original float32 tensor.
        """
        return (q_x - zero_point) * scale

if __name__ == "__main__":
    torch.manual_seed(42)
    # Verification Routine
    print("--- Quantization Precision Check ---")
    x = torch.randn(1024) * 10.0
    x[0] = 50.0 # Introduce outlier
    
    # Weights Check (Symmetric)
    q_sym = Quantizer(bits=8, symmetric=True)
    s, z = q_sym.calibrate(x)
    dx = q_sym.dequantize(q_sym.quantize(x, s, z), s, z)
    mse_sym = torch.mean((x - dx)**2)
    print(f"Symmetric (8-bit) MSE : {mse_sym:.6f}")
    
    # Activations Check (Asymmetric)
    q_asym = Quantizer(bits=8, symmetric=False)
    s, z = q_asym.calibrate(x)
    dx = q_asym.dequantize(q_asym.quantize(x, s, z), s, z)
    mse_asym = torch.mean((x - dx)**2)
    print(f"Asymmetric (8-bit) MSE: {mse_asym:.6f}")