/*
 * A custom implementation of the Ring All-Reduce algorithm using blocking MPI 
 * primitives. This demonstrates the fundamental logic behind NCCL's bandwidth 
 * optimization.
 * * Logic:
 * We sum a vector of floats across N processes.
 * topology: 0 -> 1 -> ... -> N-1 -> 0
 */

#include <mpi.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

// Helper to clean up the output
void log(int rank, std::string msg) {
    std::cout << "[Process " << rank << "] " << msg << std::endl;
}

int main(int argc, char** argv) {
    // 1. Initialize the MPI Environment
    MPI_Init(&argc, &argv);

    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // We need at least 2 processes to form a ring
    if (world_size < 2) {
        if (world_rank == 0) {
            std::cerr << "Error: This simulation requires at least 2 MPI processes." << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    // 2. Setup "Gradients"
    // In a real scenario, this would be millions of parameters.
    // Here, we simulate a model with 10 weights.
    const int DATA_SIZE = 10;
    std::vector<float> gradients(DATA_SIZE);
    
    // Fill with dummy data: Rank 0 has all 1s, Rank 1 has all 2s, etc.
    // If we have 4 processes, result should be 1+2+3+4 = 10 at every index (if rank indices were 1-based)
    // Actually using rank + 1.0f for clarity.
    std::fill(gradients.begin(), gradients.end(), (float)(world_rank + 1));

    // 3. Define Ring Topology
    // Send to (rank + 1), Receive from (rank - 1)
    int right_neighbor = (world_rank + 1) % world_size;
    int left_neighbor = (world_rank - 1 + world_size) % world_size;

    // buffer to receive incoming data from the left
    std::vector<float> recv_buffer(DATA_SIZE);

    // 4. The Ring Loop
    // For a simple All-Reduce simulation, we will pass the ENTIRE vector around 
    // the ring (N-1) times. 
    // Note: A production Ring-Reduce splits data into chunks. 
    // This is a simplified "Pass-and-Add" implementation for demonstration.
    
    for (int step = 0; step < world_size - 1; ++step) {
        
        // CRITICAL: Deadlock Prevention
        // If everyone calls MPI_Send at the same time, they all block waiting for a Recv.
        // Strategy: Even ranks Send then Recv. Odd ranks Recv then Send.
        
        MPI_Status status;

        if (world_rank % 2 == 0) {
            // Even: Send Right -> Recv Left
            // Tag 0 is just a generic tag
            MPI_Send(gradients.data(), DATA_SIZE, MPI_FLOAT, right_neighbor, 0, MPI_COMM_WORLD);
            MPI_Recv(recv_buffer.data(), DATA_SIZE, MPI_FLOAT, left_neighbor, 0, MPI_COMM_WORLD, &status);
        } else {
            // Odd: Recv Left -> Send Right
            MPI_Recv(recv_buffer.data(), DATA_SIZE, MPI_FLOAT, left_neighbor, 0, MPI_COMM_WORLD, &status);
            MPI_Send(gradients.data(), DATA_SIZE, MPI_FLOAT, right_neighbor, 0, MPI_COMM_WORLD);
        }

        // Accumulate: Add the received gradients to our own
        // In a real ring-reduce, we'd only add specific chunks.
        // Here we simulate the accumulation of knowledge.
        for (int i = 0; i < DATA_SIZE; ++i) {
            gradients[i] += recv_buffer[i];
        }
        
        // Log the step (only rank 0 to avoid terminal spam)
        if (world_rank == 0) {
             printf("Rank 0: Completed Step %d/%d\n", step + 1, world_size - 1);
        }
    }

    // 5. Verification
    // After the loop, everyone should have the sum of all ranks.
    // Sum of 1..N = N(N+1)/2. 
    // If N=4, Sum=10.
    float expected_val = (world_size * (world_size + 1)) / 2.0f;
    
    // Check index 0
    if (std::abs(gradients[0] - expected_val) < 0.001) {
        printf("Process %d: SUCCESS. Value is %.2f (Expected %.2f)\n", 
               world_rank, gradients[0], expected_val);
    } else {
        printf("Process %d: FAILURE. Value is %.2f (Expected %.2f)\n", 
               world_rank, gradients[0], expected_val);
    }

    MPI_Finalize();
    return 0;
}