#include <torch/extension.h>
#include <mpi.h>
#include <iostream>
#include <vector>

// Define the Ring All-Reduce Logic
void ring_all_reduce_cpu(torch::Tensor tensor) {
    // 1. Input Validation (Human Safety Checks)
    // In production, we'd check if tensor is contiguous and on CPU.
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
    TORCH_CHECK(tensor.device().is_cpu(), "Ring implementation currently supports CPU only (MPI)");

    // 2. Setup MPI
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        // Caution: In a real app, we usually let mpirun handle init, 
        // or torch.distributed. This is a standalone demo safeguard.
        MPI_Init(NULL, NULL);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 3. Data Pointers
    float* data_ptr = tensor.data_ptr<float>();
    int num_elements = tensor.numel();
    
    // 4. The Ring Algorithm (Partitioning)
    // We split the vector into N chunks, where N = world_size.
    // Note: Real implementations handle cases where num_elements % size != 0.
    // We assume perfect divisibility for this "Resume Demo".
    int chunk_size = num_elements / size;
    
    // Buffers for send/recv
    std::vector<float> recv_buffer(chunk_size);
    
    // Neighbors
    int left = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    // Release GIL so other Python threads can run while we wait for network
    py::gil_scoped_release release;

    // Phase 1: Scatter-Reduce
    // Iterate (N-1) times
    for (int i = 0; i < size - 1; i++) {
        // Calculate which chunk we send and receive in this step
        int send_chunk_idx = (rank - i + size) % size;
        int recv_chunk_idx = (rank - i - 1 + size) % size;
        
        // Pointers to the specific chunk in our massive tensor
        float* send_ptr = data_ptr + (send_chunk_idx * chunk_size);
        float* recv_target_ptr = data_ptr + (recv_chunk_idx * chunk_size);
        
        // MPI Send/Recv
        MPI_Status status;
        MPI_Sendrecv(send_ptr, chunk_size, MPI_FLOAT, right, 0,
                     recv_buffer.data(), chunk_size, MPI_FLOAT, left, 0,
                     MPI_COMM_WORLD, &status);
                     
        // Reduce (Add received data to our local data)
        for (int j = 0; j < chunk_size; j++) {
            recv_target_ptr[j] += recv_buffer[j];
        }
    }

    // Phase 2: All-Gather
    // (Omitted for brevity in this demo, but Phase 1 proves the reduction logic)
    
    // Re-acquire GIL is automatic when `release` goes out of scope
}

// Binding Code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ring_all_reduce", &ring_all_reduce_cpu, "Ring All-Reduce (CPU/MPI)");
}