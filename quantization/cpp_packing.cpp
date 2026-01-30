#include <torch/extension.h>
#include <vector>

// C++ Implementation of INT4 Packing
// A high-performance replacement for the Python bitwise reference implementation.
// Compiles on Apple Silicon (CPU) and Linux Clusters (CPU).

torch::Tensor pack_int4_cpu(torch::Tensor a, torch::Tensor b) {
    // 1. Safety Checks (Fail fast)
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensors must have the same shape");
    TORCH_CHECK(a.dtype() == torch::kUInt8, "Input A must be uint8");
    TORCH_CHECK(b.dtype() == torch::kUInt8, "Input B must be uint8");
    
    // Performance: Ensure continuous memory so we can iterate linearly
    TORCH_CHECK(a.is_contiguous(), "Input A must be contiguous for fast packing");
    TORCH_CHECK(b.is_contiguous(), "Input B must be contiguous for fast packing");

    // 2. Output Allocation
    auto packed = torch::empty_like(a);
    
    // 3. Raw Pointers for Speed (No Tensor accessor overhead)
    uint8_t* a_ptr = a.data_ptr<uint8_t>();
    uint8_t* b_ptr = b.data_ptr<uint8_t>();
    uint8_t* packed_ptr = packed.data_ptr<uint8_t>();
    
    int64_t numel = a.numel();

    // 4. Vectorized Loop
    // Layout: [High 4 bits: b] | [Low 4 bits: a]
    // #pragma omp parallel for // Uncomment if you enable OpenMP in setup.py
    for (int64_t i = 0; i < numel; i++) {
        // We assume inputs are already clipped to [0, 15] for speed
        packed_ptr[i] = (b_ptr[i] << 4) | (a_ptr[i] & 0x0F);
    }

    return packed;
}

std::vector<torch::Tensor> unpack_int4_cpu(torch::Tensor packed) {
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "Input must be uint8");
    TORCH_CHECK(packed.is_contiguous(), "Input must be contiguous");

    auto a = torch::empty_like(packed);
    auto b = torch::empty_like(packed);

    uint8_t* packed_ptr = packed.data_ptr<uint8_t>();
    uint8_t* a_ptr = a.data_ptr<uint8_t>();
    uint8_t* b_ptr = b.data_ptr<uint8_t>();
    
    int64_t numel = packed.numel();

    for (int64_t i = 0; i < numel; i++) {
        uint8_t val = packed_ptr[i];
        a_ptr[i] = val & 0x0F;      // Low nibble
        b_ptr[i] = (val >> 4);      // High nibble
    }

    return {a, b};
}

// Python Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_int4", &pack_int4_cpu, "Pack two int4 (uint8 container) tensors into one uint8 tensor");
    m.def("unpack_int4", &unpack_int4_cpu, "Unpack one uint8 tensor into two int4 tensors");
}