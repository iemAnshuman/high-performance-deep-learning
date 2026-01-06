import os
import torch
import torch.distributed as dist
import torch.nn as nn

class ManualDDP:
    """
    A bare-bones implementation of Distributed Data Parallel (DDP).
    
    Why write this? 
    To understand how PyTorch hooks into the backward pass to trigger 
    communication (All-Reduce) while computation is still happening.
    """
    def __init__(self, model: nn.Module, rank: int, world_size: int, verbose=False):
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.verbose = verbose
        
        # Make sure the model is on the right device
        # In a real setup, we'd ensure model is on cuda:{rank}
        self.device = next(model.parameters()).device
        
        # Register the hooks immediately
        self._register_backward_hooks()
        
        if self.verbose and self.rank == 0:
            print(f"[ManualDDP] Initialized on Rank {self.rank}. World Size: {self.world_size}")

    def _register_backward_hooks(self):
        """
        Iterate over all parameters. Attach a hook that runs 
        IMMEDIATELY after the gradient for that parameter is computed.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # We use a closure (nested function) to capture the parameter 'param'
                # cleanly. This is a common Python trick.
                def hook_wrapper(grad, p_name=name):
                    self._all_reduce_gradient(grad, p_name)
                    # We must return the grad (or modified grad)
                    return grad
                
                # Register the hook
                param.register_hook(hook_wrapper)

    def _all_reduce_gradient(self, grad_tensor, name):
        """
        The core synchronization logic.
        1. Async All-Reduce (Sum)
        2. Divide by World_Size to get the Mean
        """
        # 1. Reduction (Summing gradients from all GPUs)
        # We use async_op=False here for safety/simplicity, effectively blocking 
        # specifically for this tensor. Real DDP uses async buckets.
        dist.all_reduce(grad_tensor, op=dist.ReduceOp.SUM)
        
        # 2. Average
        grad_tensor.div_(self.world_size)
        
        if self.verbose:
            # Only print occasionally or for specific layers to avoid log spam
            if "fc" in name and self.rank == 0: 
                print(f"[Rank {self.rank}] Synced gradient for {name}")

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

# Usage Example (Simulated)
if __name__ == "__main__":
    # Standard boilerplate to setup a local distributed process
    # Usually run via: torchrun --nproc_per_node=2 distributed/manual_ddp.py
    
    # Check if we are running inside a distributed launcher
    if "RANK" in os.environ:
        dist.init_process_group(backend="gloo") # 'gloo' works on CPU, 'nccl' needs GPU
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        print("Not running in distributed mode. Exiting test.")
        exit(0)

    # 1. Define a simple model
    model = nn.Linear(10, 1)
    
    # 2. Wrap it
    ddp_model = ManualDDP(model, rank, world_size, verbose=True)

    # 3. Dummy Training Loop
    input_data = torch.randn(5, 10)
    target = torch.randn(5, 1)
    
    # Forward
    output = ddp_model.forward(input_data)
    loss = (output - target).pow(2).sum()
    
    # Backward -> This triggers the hooks!
    print(f"[Rank {rank}] Starting Backward...")
    loss.backward()
    print(f"[Rank {rank}] Backward Finished.")

    # Cleanup
    dist.destroy_process_group()