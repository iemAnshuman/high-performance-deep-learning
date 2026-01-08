import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

class ColumnParallelLinear(nn.Module):
    """
    A Linear layer that splits the Weight matrix along the COLUMN axis.
    
    Standard Linear: Y = X * W
    Parallel Linear: Y = [X * W_1, X * W_2] 
    
    Each GPU holds a slice of W. The output Y is distributed (split) 
    across GPUs. We can optionally gather it at the end.
    """
    def __init__(self, input_size, output_size, gather_output=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        
        # 1. Setup Distributed Context
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            # Fallback for local testing (Human-friendly: doesn't crash on CPU)
            self.world_size = 1
            self.rank = 0

        # 2. Divide output size by number of GPUs
        # Ensure it divides evenly (Real engineers check this!)
        assert output_size % self.world_size == 0, "Output size must be divisible by world size"
        self.output_size_per_partition = output_size // self.world_size

        # 3. Initialize Parameters
        # Note: We allocate the SMALLER (sharded) matrix to save memory.
        self.weight = nn.Parameter(torch.empty(
            self.output_size_per_partition, 
            self.input_size
        ))
        
        # Helper to visualize what's happening
        if self.rank == 0:
            print(f"[TP] Initialized ColumnLinear. Logical: {output_size} -> Physical: {self.output_size_per_partition} per GPU")

        self.reset_parameters()

    def reset_parameters(self):
        """
        Crucial: In a real training run, we want the full conceptual matrix W 
        to be initialized consistently, then split. 
        Here, for simplicity, we just initialize the shards randomly.
        """
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input_tensor):
        # 1. Local MatMul
        # Input: [Batch, Input_Size] (Replicated on all GPUs)
        # Weight: [Output_Size_Part, Input_Size] (Unique per GPU)
        # Result: [Batch, Output_Size_Part]
        local_output = F.linear(input_tensor, self.weight)

        # 2. Gather (Optional)
        # If this is the final layer, we usually want to assemble the full result.
        if self.gather_output and self.world_size > 1:
            return self._gather_all(local_output)
        
        return local_output

    def _gather_all(self, local_output):
        # Prepare a list of tensors to catch the incoming data
        gathered_tensors = [torch.zeros_like(local_output) for _ in range(self.world_size)]
        
        # All-Gather: Everyone sends their piece to everyone else
        dist.all_gather(gathered_tensors, local_output)
        
        # Concatenate along the last dimension to reconstruct the full vector
        # [Part1, Part2, Part3...] -> Full Output
        full_output = torch.cat(gathered_tensors, dim=-1)
        
        return full_output

# Usage Example
if __name__ == "__main__":
    # Simulate a run
    print("Running Tensor Parallel Linear Test...")
    
    # Fake distributed setup
    # In reality, I'd run this with torchrun <-- for demo
    layer = ColumnParallelLinear(input_size=1024, output_size=4096, gather_output=True)
    
    dummy_input = torch.randn(2, 1024)
    output = layer(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}") 
    # If world_size was 1, Output should be [2, 4096]
    # If world_size was 2, Local would be [2, 2048], Gathered would be [2, 4096]