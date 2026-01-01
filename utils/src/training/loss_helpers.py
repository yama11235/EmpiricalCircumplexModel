"""Helper utilities for loss computation."""
from typing import Optional
import torch


def accumulate_loss(
    current: Optional[torch.Tensor],
    new_loss: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """
    Accumulate losses.
    
    Args:
        current: Current accumulated loss
        new_loss: New loss to add
        
    Returns:
        Accumulated loss
    """
    if new_loss is None:
        return current
    if current is None:
        return new_loss
    return current + new_loss

