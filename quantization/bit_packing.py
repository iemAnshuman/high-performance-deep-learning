import torch
import numpy as np

def pack_int4(u8_tensor_a, u8_tensor_b):
    """
    Packs two int4 tensors (stored as uint8) into one uint8 tensor.
    
    Layout:
    |  4 bits (b) | 4 bits (a) |
    
    Args:
        u8_tensor_a: Tensor containing values [0, 15]
        u8_tensor_b: Tensor containing values [0, 15]
    """
    # Defensive: Ensure inputs are actually in 4-bit range
    # In a real kernel, we'd skip this check for speed, but this is a util.
    assert (u8_tensor_a.max() < 16) and (u8_tensor_b.max() < 16), "Inputs must be int4"
    
    # Shift 'b' to the upper nibble and OR with 'a'
    packed = (u8_tensor_b << 4) | (u8_tensor_a & 0x0F)
    return packed.to(torch.uint8)

def unpack_int4(packed_tensor):
    """
    Unpacks a uint8 tensor into two int4 tensors.
    """
    # Mask out the upper bits to get 'a'
    # 0x0F is 00001111 in binary
    a = packed_tensor & 0x0F
    
    # Shift down to get 'b'
    b = packed_tensor >> 4
    
    return a, b

if __name__ == "__main__":
    # Test Data: Random integers between 0 and 15
    a = torch.randint(0, 16, (10,), dtype=torch.uint8)
    b = torch.randint(0, 16, (10,), dtype=torch.uint8)
    
    print(f"Original A: {a}")
    print(f"Original B: {b}")
    
    # Pack
    packed = pack_int4(a, b)
    print(f"Packed (Hex): {[hex(x.item()) for x in packed]}")
    
    # Unpack
    ua, ub = unpack_int4(packed)
    
    if torch.all(ua == a) and torch.all(ub == b):
        print("Success! Bit packing/unpacking is lossless.")
    else:
        print("Failure! Data corruption detected.")
