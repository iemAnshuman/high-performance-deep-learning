# Compiler settings
CXX = mpicxx
NVCC = nvcc
PYTHON = python3

# Flags
CXX_FLAGS = -O3 -Wall -std=c++17
NVCC_FLAGS = -O3 --ptxas-options=-v -arch=sm_80

# Directories
BUILD_DIR = build
SRC_DIR = distributed

# Targets
.PHONY: all clean run_ring

all: directories ring_reduce

directories:
	mkdir -p $(BUILD_DIR)

# 1. Compile the MPI Ring Reduce (CPU/MPI)
ring_reduce: $(SRC_DIR)/ring_reduce.cpp
	@echo "Compiling Ring All-Reduce..."
	$(CXX) $(CXX_FLAGS) $< -o $(BUILD_DIR)/ring_reduce

# 2. (Optional) If we had a CUDA kernel, we would compile it like this:
# This shows intent to expand the project later.
# triton_kernel: kernels/fused_attention.cu
# 	$(NVCC) $(NVCC_FLAGS) $< -o $(BUILD_DIR)/triton_kernel

# 3. Utility: Run the ring simulation
run_ring: ring_reduce
	@echo "Running simulation with 4 processes..."
	mpirun -np 4 ./$(BUILD_DIR)/ring_reduce

clean:
	rm -rf $(BUILD_DIR)
	rm -f *.pyc
	rm -rf __pycache__