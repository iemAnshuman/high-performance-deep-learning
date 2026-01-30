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
        self.offsets = {}     # Map param -> (start, end) in buffer
        self.ready_count = 0 
        self.device = device
        self.buffer = None 

    def add_param(self, name, param):
        self.params.append((name, param))
        num_bytes = param.numel() * param.element_size()
        
        # Track offset for this param
        start = self.current_bytes // 4 # float32 indices
        end = start + param.numel()
        self.offsets[param] = (start, end)
        
        self.current_bytes += num_bytes

    def is_full(self):
        return self.current_bytes >= self.max_bytes

    def finalize_buffer(self):
        if self.current_bytes == 0:
            return
        # Create flat buffer
        self.buffer = torch.zeros(
            self.current_bytes // 4, 
            dtype=torch.float32, 
            device=self.device
        )

    def reset(self):
        self.ready_count = 0
        self.buffer.zero_()

class BucketedDDP(nn.Module):
    def __init__(self, model, bucket_cap_mb=25):
        super().__init__()
        self.model = model
        
        # 1. Setup Distributed
        if not dist.is_initialized():
            raise RuntimeError("BucketedDDP requires torch.distributed!")
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = next(model.parameters()).device
        
        # 2. Assign Parameters to Buckets (Reverse Order)
        self.buckets = []
        current_bucket = Bucket(0, bucket_cap_mb, self.device)
        
        # Filter only trainable params
        params = [p for p in model.named_parameters() if p[1].requires_grad]
        
        for name, param in reversed(params):
            if current_bucket.is_full():
                current_bucket.finalize_buffer()
                self.buckets.append(current_bucket)
                current_bucket = Bucket(len(self.buckets), bucket_cap_mb, self.device)
            
            current_bucket.add_param(name, param)
            
            if not hasattr(self, 'param_to_bucket'):
                self.param_to_bucket = {}
            self.param_to_bucket[param] = current_bucket

        if current_bucket.current_bytes > 0:
            current_bucket.finalize_buffer()
            self.buckets.append(current_bucket)

        # 3. Register Hooks
        self._register_hooks()

    def _register_hooks(self):
        for param in self.model.parameters():
            if param.requires_grad:
                # We do NOT use param.retain_grad() here necessarily, 
                # but we rely on the hook argument.
                def hook(grad, p=param):
                    self._on_grad_ready(p, grad)
                    return grad
                param.register_hook(hook)

    def _on_grad_ready(self, param, grad):
        bucket = self.param_to_bucket[param]
        
        # FIX: Copy gradient into buffer IMMEDIATELY using the stored offset
        # We don't wait for _sync_bucket because 'grad' is transient
        start, end = bucket.offsets[param]
        bucket.buffer[start:end] = grad.view(-1)
        
        bucket.ready_count += 1
        
        if bucket.ready_count == len(bucket.params):
            self._sync_bucket(bucket)

    def _sync_bucket(self, bucket):
        # 1. Reduce: The expensive network call
        dist.all_reduce(bucket.buffer, op=dist.ReduceOp.SUM)
        
        # 2. Average
        if self.world_size > 1:
            bucket.buffer.div_(self.world_size)
        
        # 3. Unpack: Write back to params
        # Note: Since hooks for previous params already finished, we update 
        # param.grad manually. This is a simplification for the demo.
        for name, param in bucket.params:
            start, end = bucket.offsets[param]
            synced_grad = bucket.buffer[start:end].view(param.shape)
            
            # If param.grad is None (likely), we create it. 
            # If it exists, we overwrite it.
            if param.grad is None:
                param.grad = synced_grad.clone()
            else:
                param.grad.copy_(synced_grad)
            
        bucket.reset()

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)