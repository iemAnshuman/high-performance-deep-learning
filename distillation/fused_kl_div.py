import torch
import triton
import triton.language as tl

@triton.jit
def fused_kl_kernel(
    teacher_ptr, student_ptr, loss_ptr,
    stride_row, 
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """
    Computes KL Divergence inline without materializing probabilities.
    Input: Logits (Not Softmaxed)
    Output: Scalar Loss
    """
    row_idx = tl.program_id(0)
    
    # Pointers to the start of the row
    row_start_t = teacher_ptr + row_idx * stride_row
    row_start_s = student_ptr + row_idx * stride_row
    
    # Load the entire row (Vocab size)
    # In production, we'd need to tile this loop if Vocab > SRAM size.
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Load Logits
    t_logits = tl.load(row_start_t + offsets, mask=mask, other=-float('inf'))
    s_logits = tl.load(row_start_s + offsets, mask=mask, other=-float('inf'))

    # 1. Softmax (Teacher)
    # Numerical stability: subtract max
    t_max = tl.max(t_logits, axis=0)
    t_sub = t_logits - t_max
    t_exp = tl.exp(t_sub)
    t_sum = tl.sum(t_exp, axis=0)
    # Log_Softmax = (x - max) - log(sum)
    t_log_probs = t_sub - tl.log(t_sum)
    t_probs = t_exp / t_sum

    # 2. Softmax (Student)
    s_max = tl.max(s_logits, axis=0)
    s_sub = s_logits - s_max
    s_exp = tl.exp(s_sub)
    s_sum = tl.sum(s_exp, axis=0)
    s_log_probs = s_sub - tl.log(s_sum)

    # 3. KL Divergence Formula: sum( P(x) * (logP(x) - logQ(x)) )
    kl_pointwise = t_probs * (t_log_probs - s_log_probs)
    
    # Sum over the row
    kl_sum = tl.sum(kl_pointwise, axis=0)

    # Write back result
    tl.store(loss_ptr + row_idx, kl_sum)

def fused_kl_loss(teacher_logits, student_logits):
    n_rows, n_cols = teacher_logits.shape
    # For simulation, assume vocab fits in block
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    losses = torch.empty(n_rows, device=teacher_logits.device, dtype=torch.float32)
    
    grid = (n_rows,)
    fused_kl_kernel[grid](
        teacher_logits, student_logits, losses,
        teacher_logits.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return losses.mean()

# Correctness Check
if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.manual_seed(0)
        # Small dimensions for testing
        T = torch.randn(32, 1024, device='cuda')
        S = torch.randn(32, 1024, device='cuda')
        
        # 1. Custom Triton Op
        custom_loss = fused_kl_loss(T, S)
        
        # 2. Standard PyTorch Op
        # F.kl_div expects input to be log-probs, target to be probs (usually)
        # depending on reduction. Doing manual for clarity:
        T_p = torch.softmax(T, dim=1)
        T_log = torch.log_softmax(T, dim=1)
        S_log = torch.log_softmax(S, dim=1)
        torch_loss = torch.sum(T_p * (T_log - S_log), dim=1).mean()
        
        print(f"Triton Loss: {custom_loss.item():.4f}")
        print(f"PyTorch Loss: {torch_loss.item():.4f}")