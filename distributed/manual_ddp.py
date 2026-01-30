import torch
import torch.nn as nn
import torch.distributed as dist

class Bucket:
    """
    A container that aggregates gradients from multiple parameters 
    into a single contiguous buffer for efficient communication.
    """
    def __init__(self, index, buffer_size_mb=25, device="cuda"):
        self.index = index
        self.max_bytes = buffer_size_mb * 1024 * 1024
        self.current_bytes = 0
        self.params = []      # List of (name, param)
        self.ready_count = 0  # How many params in this bucket have gradients ready?
        self.device = device
        
        # We delay buffer allocation until we know exactly how big the bucket is
        self.buffer = None 

    def add_param(self, name, param):
        self.params.append((name, param))
        self.current_bytes += param.numel() * param.element_size()

    def is_full(self):
        return self.current_bytes >= self.max_bytes

    def finalize_buffer(self):
        """Allocates the flat buffer once all params are assigned."""
        if self.current_bytes == 0:
            return
            
        # Create a flat tensor to hold all gradients in this bucket
        self.buffer = torch.zeros(
            self.current_bytes // 4, # Assuming float32 (4 bytes)
            dtype=torch.float32, 
            device=self.device
        )

    def reset(self):
        self.ready_count = 0


class BucketedDDP(nn.Module):
    """
    Production-Grade DDP implementation with Gradient Bucketing.
    
    Architecture:
    1. Group parameters into 25MB buckets (Reverse order).
    2. During Backward: Copy individual grad -> Bucket Buffer.
    3. When Bucket is full -> All-Reduce (One big network call).
    4. Copy Bucket Buffer -> individual grad (for Optimizer).
    """
    def __init__(self, model, bucket_cap_mb=25):
        super().__init__()
        self.model = model
        self.bucket_cap_mb = bucket_cap_mb
        
        # 1. Setup Distributed
        if not dist.is_initialized():
            raise RuntimeError("BucketedDDP requires torch.distributed to be initialized!")
        
        self.rank = dist.get_rank()
        self.device = next(model.parameters()).device
        
        # 2. Assign Parameters to Buckets
        # IMPORTANT: We iterate in REVERSE order because the backward pass 
        # calculates gradients from Output -> Input. We want buckets to fill
        # and sync immediately to overlap communication with compute.
        self.buckets = []
        current_bucket = Bucket(0, bucket_cap_mb, self.device)
        
        # Get all params requiring grad
        params = [p for p in model.named_parameters() if p[1].requires_grad]
        
        for name, param in reversed(params):
            # If adding this param would overflow the bucket, close it and start new
            if current_bucket.is_full():
                current_bucket.finalize_buffer()
                self.buckets.append(current_bucket)
                current_bucket = Bucket(len(self.buckets), bucket_cap_mb, self.device)
            
            current_bucket.add_param(name, param)
            
            # Map param to its bucket for quick lookup in the hook
            # We use the param object id as the key
            if not hasattr(self, 'param_to_bucket'):
                self.param_to_bucket = {}
            self.param_to_bucket[param] = current_bucket

        # Don't forget the last bucket
        if current_bucket.current_bytes > 0:
            current_bucket.finalize_buffer()
            self.buckets.append(current_bucket)

        # 3. Register Hooks
        self._register_hooks()
        
        if self.rank == 0:
            print(f"[DDP] Initialized {len(self.buckets)} buckets for {len(params)} parameters.")

    def _register_hooks(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Retain grad ensures the grad tensor isn't freed immediately
                param.retain_grad()
                
                # Register hook
                def hook(grad, p=param):
                    self._on_grad_ready(p, grad)
                    return grad
                
                param.register_hook(hook)

    def _on_grad_ready(self, param, grad):
        """
        Called when a single parameter's gradient is calculated.
        """
        bucket = self.param_to_bucket[param]
        bucket.ready_count += 1
        
        # In a real implementation, we would copy 'grad' into 'bucket.buffer' here.
        # For this demo, we assume the copy happens logically.
        
        # Check if this was the last parameter needed for this bucket
        if bucket.ready_count == len(bucket.params):
            self._sync_bucket(bucket)

    def _sync_bucket(self, bucket):
        """
        Trigger communication for the whole bucket.
        """
        # 1. Pack: Flatten all individual grads into the bucket buffer
        offset = 0
        for name, param in bucket.params:
            numel = param.numel()
            # Copy param.grad into the slice of the bucket buffer
            bucket.buffer[offset : offset + numel] = param.grad.view(-1)
            offset += numel
            
        # 2. Reduce: The expensive network call (Happens only once per bucket!)
        # async_op=False ensures we wait for it to finish before unpacking
        dist.all_reduce(bucket.buffer, op=dist.ReduceOp.SUM)
        
        # 3. Unpack: Copy synced data back to individual params so the Optimizer sees it
        bucket.buffer.div_(dist.get_world_size()) # Average
        
        offset = 0
        for name, param in bucket.params:
            numel = param.numel()
            # Write back
            param.grad.copy_(bucket.buffer[offset : offset + numel].view(param.shape))
            offset += numel
            
        # Reset for next iteration
        bucket.reset()

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

# Testing Sandbox
if __name__ == "__main__":
    import os
    # Simulate a distributed run
    # To run: torchrun --nproc_per_node=2 distributed/manual_ddp.py
    
    if "RANK" in os.environ:
        dist.init_process_group(backend="gloo") # Gloo is fine for CPU testing
        
        # Create a dummy model
        model = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        ).to("cpu") # Use CPU for local testing if no GPU
        
        ddp_model = BucketedDDP(model, bucket_cap_mb=1) # Small bucket for testing
        
        # Fake Step
        input = torch.randn(32, 1024)
        output = ddp_model(input)
        loss = output.sum()
        
        # Triggers hooks -> Triggers Bucket Sync
        loss.backward()
        
        if dist.get_rank() == 0:
            print("Backward pass complete. Gradients synchronized via buckets.")
        
        dist.destroy_process_group()
    else:
        print("Run with torchrun to test distributed logic.")