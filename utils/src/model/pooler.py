"""Pooling utilities for sentence embeddings."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["Pooler"]


class Pooler(nn.Module):
    """Parameter-free poolers to get the sentence embedding."""

    def __init__(self, pooler_type: str):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            "cls",
            "avg",
            "last",
            "max",
        ], f"unrecognized pooling type {self.pooler_type}"

    def forward(self, attention_mask: torch.Tensor, outputs, target_layer: int = -1):
        if isinstance(outputs, torch.Tensor):
            last_hidden = outputs
            hidden_states = (outputs,)
        else:
            last_hidden = outputs.last_hidden_state
            hidden_states = outputs.hidden_states or (last_hidden,)

        dtype = last_hidden.dtype

        if self.pooler_type in "cls":
            return hidden_states[target_layer][:, 0].to(dtype)
        if self.pooler_type == "avg":
            return (
                (hidden_states[target_layer] * attention_mask.unsqueeze(-1)).sum(1)
                / attention_mask.sum(-1).unsqueeze(-1)
            ).to(dtype)
        if self.pooler_type == "max":
            mask = attention_mask.unsqueeze(-1).bool()
            masked_hidden = hidden_states[target_layer].masked_fill(
                ~mask, torch.finfo(hidden_states[target_layer].dtype).min
            )
            pooled_result, _ = masked_hidden.max(dim=1)
            return pooled_result.to(dtype)
        if self.pooler_type == "last":
            lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(attention_mask.size(0), device=attention_mask.device)
            return hidden_states[target_layer][batch_idx, lengths].to(dtype)

        raise NotImplementedError
