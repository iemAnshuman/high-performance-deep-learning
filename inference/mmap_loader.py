import torch
import mmap
import os
import time
import contextlib

class ZeroCopyLoader:
    """
    Real Zero-Copy Loader using torch.frombuffer.
    
    This allows loading models larger than RAM by leveraging the OS page cache.
    The tensor is created instantly; data is faulted in by the OS on access.
    """
    def __init__(self, filename):
        self.filename = filename
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file {filename} not found.")
            
        # Keep file open as long as the mmap is active
        self._file_handle = open(self.filename, "r+b")
        
        # Create the memory map
        # PROT_READ means the OS ensures we don't accidentally overwrite weights
        self.mm = mmap.mmap(self._file_handle.fileno(), 0, access=mmap.ACCESS_READ)

    def load_tensor(self, offset, shape, dtype=torch.float32):
        """
        Returns a torch.Tensor view of the file without copying bytes to userspace RAM.
        """
        # 1. Calculate size
        numel = 1
        for dim in shape:
            numel *= dim
            
        # 2. Create the Tensor View
        # torch.frombuffer creates a tensor that shares memory with the python object (self.mm)
        # No memory copy happens here. It's just pointer arithmetic.
        try:
            flat_tensor = torch.frombuffer(
                self.mm, 
                dtype=dtype, 
                count=numel, 
                offset=offset
            )
        except Exception as e:
            print(f"[Error] Failed to map tensor: {e}")
            return None

        # 3. Reshape (View)
        return flat_tensor.view(shape)

    def close(self):
        """Clean up file handles"""
        if hasattr(self, 'mm') and not self.mm.closed:
            self.mm.close()
        if hasattr(self, '_file_handle') and not self._file_handle.closed:
            self._file_handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Usage Simulation
if __name__ == "__main__":
    # 1. Generate a dummy 1GB file
    dummy_file = "large_model.bin"
    size_in_floats = 256 * 1024 * 1024 # 1GB of float32
    
    if not os.path.exists(dummy_file):
        print("Generating 1GB dummy file...")
        with open(dummy_file, "wb") as f:
            f.seek((size_in_floats * 4) - 1)
            f.write(b'\0')

    print("--- Testing Zero-Copy Load ---")
    
    # 2. The "Instant" Load
    t0 = time.time()
    with ZeroCopyLoader(dummy_file) as loader:
        # Map a chunk in the middle of the file
        # This returns a real PyTorch tensor
        tensor = loader.load_tensor(offset=1024, shape=(4096, 4096), dtype=torch.float32)
        
        load_time = time.time() - t0
        print(f"Load Time: {load_time*1000:.3f} ms (Should be < 1ms)")
        
        # 3. Prove it's usable
        print(f"Tensor Device: {tensor.device}")
        print(f"Tensor Shape:  {tensor.shape}")
        
        # 4. Trigger the Page Fault (The actual read from disk happens here)
        t1 = time.time()
        s = torch.sum(tensor) 
        access_time = time.time() - t1
        print(f"Access Time: {access_time*1000:.3f} ms (Disk I/O happened here)")

    # Cleanup
    if os.path.exists(dummy_file):
        os.remove(dummy_file)