"""
Loss function computation - single sentence only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch

from utils.src.training.loss_helpers import (
    accumulate_loss,
)

TensorDict = Dict[str, torch.Tensor]
LossResult = Tuple[Optional[torch.Tensor], TensorDict]


def compute_single_loss(
    trainer: Any,
    model: torch.nn.Module,
    inputs: Optional[Dict[str, Any]],
    active_heads: List[str],
    device: torch.device,
) -> LossResult:
    """Compute loss for single sentence tasks."""
    if not inputs:
        return None, {}

    outputs = model(**inputs)
    loss: Optional[torch.Tensor] = None
    labels = inputs.get("labels")

    # Convert dataset columns to classifier head names
    classifier_heads = []
    for col in active_heads:
        heads_for_col = trainer.column_to_heads.get(col, [col])
        classifier_heads.extend(heads_for_col)

    for head in classifier_heads:
        objective = trainer.head_objectives.get(head)
        assert objective is not None, f"Objective for head '{head}' is not defined."
        head_output = outputs.get(head)
        if head_output is None:
            raise KeyError(f"Model outputs missing head '{head}'")
        head_loss = objective.compute_single(trainer, head_output, labels)
        loss = accumulate_loss(loss, head_loss)
        
    return loss, outputs

__all__ = [
    "compute_single_loss",
]
