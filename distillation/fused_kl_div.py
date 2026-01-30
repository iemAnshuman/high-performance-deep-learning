import torch
import torch.nn.functional as F

# If you have a real Triton kernel for KL, put it here.
# For now, we wrap the PyTorch implementation to ensure the pipeline runs 
# even if the Triton kernel isn't fully implemented for backward.

class FusedKLDivFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, teacher_logits, student_logits):
        ctx.save_for_backward(teacher_logits, student_logits)
        
        # Use PyTorch implementation for now to guarantee correctness in demo
        # (Replacing a complex reduction kernel for stability)
        T_log_probs = F.log_softmax(teacher_logits, dim=-1)
        S_log_probs = F.log_softmax(student_logits, dim=-1)
        loss = F.kl_div(S_log_probs, T_log_probs, reduction='batchmean', log_target=True)
        
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        teacher_logits, student_logits = ctx.saved_tensors
        
        with torch.enable_grad():
            t_temp = teacher_logits.detach().requires_grad_(True)
            s_temp = student_logits.detach().requires_grad_(True)
            
            T_log = F.log_softmax(t_temp, dim=-1)
            S_log = F.log_softmax(s_temp, dim=-1)
            loss = F.kl_div(S_log, T_log, reduction='batchmean', log_target=True)
            
            loss.backward(grad_output)
            
        return t_temp.grad, s_temp.grad

def fused_kl_loss(teacher_logits, student_logits):
    return FusedKLDivFunction.apply(teacher_logits, student_logits)