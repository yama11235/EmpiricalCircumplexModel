import torch
import torch.nn.functional as F
import math
from typing import Optional, Dict

def compute_circular_cse_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    angle_map: Dict[int, str],
    id2label: Optional[Dict[int, str]] = None,
    margin: float = 0.0,
    debug: bool = False,  # Add debug flag
) -> Optional[torch.Tensor]:
    """
    Vectorized CircularCSE loss computation (Corrected Version).
    """
    if embeddings.size(0) < 2:
        return None

    device = embeddings.device
    
    # --- 1. Preprocessing ---
    z = F.normalize(embeddings.float(), dim=1)
    lbls = labels.to(device=device, dtype=torch.long).view(-1)
    
    # Build label_name_to_angle mapping
    label_name_to_angle = {name: float(ang) for ang, name in angle_map.items()}
    
    if id2label is None:
        raise ValueError(
            "id2label must be provided to compute_circular_cse_loss. "
            "Please ensure label_name_mappings is correctly passed to the trainer."
        )
    
    current_id2label = id2label
    
    if debug:
        print("\n=== CircularCSE Debug Info ===")
        print(f"id2label provided: {id2label is not None}")
        print(f"current_id2label: {current_id2label}")
        print(f"angle_map: {angle_map}")
        print(f"label_name_to_angle: {label_name_to_angle}")
        print(f"Unique label IDs in batch: {torch.unique(lbls).tolist()}")
        print("="*30 + "\n")

    valid_mask_list = []
    angles_list = []
    
    for lvl_id in lbls.tolist():
        name = current_id2label.get(lvl_id)
        if name in label_name_to_angle:
            valid_mask_list.append(True)
            angles_list.append(label_name_to_angle[name])
        else:
            valid_mask_list.append(False)
            angles_list.append(0.0)

    valid_mask = torch.tensor(valid_mask_list, device=device, dtype=torch.bool)
    # None if too few valid data
    if valid_mask.sum() < 2:
        return None

    z_valid = z[valid_mask]
    lbls_valid = lbls[valid_mask]
    angles_valid = torch.tensor(angles_list, device=device, dtype=torch.float32)[valid_mask]
    
    # Recommendation 2: Denominator of normalization should be "number of valid data"
    N_valid = z_valid.size(0)

    # --- 2. Matrix operations ---
    sim_matrix = z_valid @ z_valid.t()
    
    angle_diff = (angles_valid.unsqueeze(1) - angles_valid.unsqueeze(0)).abs()
    angle_diff = torch.min(angle_diff, 360.0 - angle_diff)
    
    angle_diff_rad = angle_diff * (math.pi / 180.0)
    target_cos_matrix = torch.cos(angle_diff_rad)
    
    same_label_matrix = (lbls_valid.unsqueeze(1) == lbls_valid.unsqueeze(0))

    # --- 3. Batch Loss Calculation (Recommendation 1: MSE) ---
    
    # First calculate pure distance (difference)
    diff = sim_matrix - target_cos_matrix
    dist = torch.abs(diff)
    
    # For Same Label:
    # Loss=0 if within Margin, otherwise squared error
    # ReLU( |x| - margin ) ^ 2
    loss_same = F.relu(dist - margin).pow(2)
    
    # For Diff Label:
    # Simple squared error (diff ^ 2)
    loss_diff = dist.pow(2)
    
    full_loss_matrix = torch.where(same_label_matrix, loss_same, loss_diff)
    
    # --- 4. Masking and Sampling ---
    triu_mask = torch.triu(torch.ones(N_valid, N_valid, device=device, dtype=torch.bool), diagonal=1)
    
    if not triu_mask.any():
        return None

    valid_losses = full_loss_matrix[triu_mask]
    
    final_losses = valid_losses
    return final_losses.mean()