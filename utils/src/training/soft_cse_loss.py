import torch
import torch.nn.functional as F
from typing import Optional, List

def compute_softcse_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    tau: float,
    pairwise_sim: torch.Tensor,
    log_of_sum: bool = False,
) -> Optional[torch.Tensor]:
    """
    SoftCSE (weight version) loss based on InfoNCE.

    Version of SoftCSE (weight) from the paper applied to labeled batches.
    - Multiply each negative example in the denominator by weight w_ij
    - w_ij is calculated based on the "supervised similarity s_ij" between anchor and negative example
      * Here, if pairwise_sim is not specified,
        consider the cos similarity of the current embedding as s_ij
      * pairwise_sim is softmax normalized after excluding self (row sum is 1)

    Args:
        embeddings: [B, D] Embeddings (trainable)
        labels:     [B]    Labels (same label => positive pair)
        tau:        float  Global temperature
        pairwise_sim: [B, B] Supervised similarity matrix (optional)
                       If None, use cos similarity of embeddings,
                       linearly transform [-1,1] -> [0,1] and use
                       In either case, it is softmax normalized

    Returns:
        loss (scalar) or None
    """
    if embeddings.size(0) < 2:
        return None

    device = embeddings.device
    eps = 1e-8

    # L2 normalized embeddings
    z = F.normalize(embeddings.float(), dim=1)
    B = z.size(0)

    lbls = labels.to(device=device, dtype=torch.long).view(-1)

    # Cosine similarity to be learned
    sim = z @ z.t()  # [B, B]
    logits = sim / max(tau, eps)

    # Mask definition (defined before supervised similarity calculation)
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    pos_mask = (lbls.unsqueeze(0) == lbls.unsqueeze(1)) & (~self_mask)
    neg_mask = (~pos_mask) & (~self_mask)

    s = pairwise_sim.to(device=device, dtype=torch.float)
    
    if not pos_mask.any():
        return None

    # Construct SoftCSE weight matrix w_ij
    # - Positive: w = 1
    # - Negative: w ~ 1 - s_ij (larger for distant negatives)
    #  Then, normalize so that the average of negatives for each anchor i is 1

    w = torch.zeros_like(sim, dtype=torch.float, device=device)  # [B, B]

    # Raw negative weights (1 - s)
    w[neg_mask] = 1.0 - s[neg_mask]

    # Scale to "average 1" for each row
    neg_counts = neg_mask.sum(dim=1, keepdim=True).clamp_min(1)
    sum_w_per_row = (w * neg_mask).sum(dim=1, keepdim=True).clamp_min(eps)
    scale = neg_counts / sum_w_per_row  # => average w ≈ 1
    w = w * scale

    # Positive weights are 1
    w = torch.where(pos_mask, torch.ones_like(w), w)

    # Self is 0 (excluded from denominator)
    w = torch.where(self_mask, torch.zeros_like(w), w)

    if log_of_sum:
        # InfoNCE denominator: Σ_j w_ij * exp(logits_ij)
        exp_logits = torch.exp(logits)
        weighted_exp = exp_logits * w  # [B, B]

        denom = weighted_exp.sum(dim=1, keepdim=True).clamp_min(eps)

        # Numerator is "positive only (weight 1)" as usual
        numer = (exp_logits * pos_mask).sum(dim=1, keepdim=True).clamp_min(eps)

        # Valid only for rows with positives
        valid = pos_mask.any(dim=1)
        if not valid.any():
            return None

        log_prob = torch.log(numer) - torch.log(denom)
        return -(log_prob.squeeze(1)[valid]).mean()
    
    else:
        # sum of log
        exp_logits = torch.exp(logits)
        weighted_exp_neg = (exp_logits * w * neg_mask).sum(dim=1, keepdim=True)

        denom = exp_logits + weighted_exp_neg

        log_prob = logits - torch.log(denom)
        valid = pos_mask.any(dim=1)
        if not valid.any():
            return None

        pos_mask_valid = pos_mask[valid]
        log_prob_valid = log_prob[valid]

        mean_log_prob_pos = (log_prob_valid * pos_mask_valid).sum(dim=1) / pos_mask_valid.sum(dim=1)
        return -mean_log_prob_pos.mean()
