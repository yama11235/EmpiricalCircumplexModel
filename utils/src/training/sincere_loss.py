"""SINCERE loss computation."""
from typing import Optional, List
import torch
import torch.nn.functional as F


def compute_sincere_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    tau: float,
    log_of_sum: bool = False,
) -> Optional[torch.Tensor]:
    """
    Compute SINCERE (contrastive) loss.
    
    Args:
        embeddings: Embedding vectors [B, D]
        labels: Labels [B]
        tau: Temperature parameter
        
    Returns:
        Loss tensor or None
    """
    if embeddings.size(0) < 2:
        return None

    device = embeddings.device
    eps = 1e-8

    # Normalize embeddings
    z = F.normalize(embeddings.float(), dim=1)    
    
    B = z.size(0)

    lbls = labels.to(device=device, dtype=torch.long).view(-1)

    # Compute similarity matrix
    sim = z @ z.t()
    logits = sim / max(tau, eps)

    # Create masks
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    pos_mask = (lbls.unsqueeze(0) == lbls.unsqueeze(1)) & (~self_mask)
    neg_mask = (~pos_mask) & (~self_mask)

    if not pos_mask.any():
        return None

    if log_of_sum:
        exp_logits = torch.exp(logits) * (~self_mask)
        denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(eps)
        numer = (exp_logits * pos_mask).sum(dim=1, keepdim=True)
        valid = pos_mask.any(dim=1)
        if not valid.any():
            return None
        log_prob = torch.log(numer.clamp_min(eps)) - torch.log(denom)
        return -(log_prob.squeeze(1)[valid]).mean()
    else:
        # sum of log
        exp_logits = torch.exp(logits) * (~self_mask)
        sum_exp_neg = (exp_logits * neg_mask).sum(dim=1, keepdim=True)
        denom = exp_logits + sum_exp_neg
        log_prob = logits - torch.log(denom)

        valid = pos_mask.any(dim=1)
        if not valid.any():
            return None

        pos_mask_valid = pos_mask[valid]
        log_prob_valid = log_prob[valid]
        
        mean_log_prob_pos = (log_prob_valid * pos_mask_valid).sum(dim=1) / pos_mask_valid.sum(dim=1)
        
        return -mean_log_prob_pos.mean()

