"""Batch processing utilities for sentence data."""
import torch
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from transformers import PreTrainedTokenizerBase


def extract_unique_strings(
    data: Union[List[str], List[List[str]]]
) -> List[str]:
    """
    Flatten List[str] or List[List[str]] once and preserve the first
    occurrence order while removing duplicates.
    """
    flattened: List[str] = []
    for elem in data:
        if isinstance(elem, list):
            flattened.extend(elem)
        else:
            flattened.append(elem)
    return list(dict.fromkeys(flattened))


def flatten_strings(
    data: Union[List[str], List[List[str]]]
) -> List[str]:
    """
    Collapse a nested string list (only one level deep) into a flat list.
    """
    if not data:
        return []
    if all(isinstance(elem, str) for elem in data):
        return data  # type: ignore
    flattened: List[str] = []
    for elem in data:
        if isinstance(elem, list):
            flattened.extend(elem)
        else:
            flattened.append(elem)
    return flattened


# ---------------------------------------------------------------------------
# Partition utilities
# ---------------------------------------------------------------------------

def compute_sentence_partitions(
    attention_mask: torch.Tensor,
    attention_mask_2: Optional[torch.Tensor] = None,
    attention_mask_3: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Inspect the presence of 1/2/3 sentence inputs inside a batch and
    return boolean masks and sample indices for each case.
    """
    if attention_mask is None:
        raise ValueError("attention_mask is required to detect batch size.")

    device = attention_mask.device
    batch_size = attention_mask.size(0)

    def _has_sentence(mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        return mask.sum(dim=1) > 0

    has_second = _has_sentence(attention_mask_2)
    has_third = _has_sentence(attention_mask_3)
    has_only_first = (~has_second) & (~has_third)

    return {
        "has_second": has_second,
        "has_third": has_third,
        "has_only_first": has_only_first,
        "idx_single": torch.nonzero(has_only_first, as_tuple=True)[0],
        "idx_pair": torch.nonzero(has_second & ~has_third, as_tuple=True)[0],
        "idx_triplet": torch.nonzero(has_third, as_tuple=True)[0],
    }


class BatchPartitioner:
    """Utility object that centralises the 1/2/3 sentence partition logic."""

    _KIND_TO_KEY = {
        "single": "idx_single",
        "pair": "idx_pair",
        "triplet": "idx_triplet",
    }

    def __init__(
        self,
        attention_mask: torch.Tensor,
        attention_mask_2: Optional[torch.Tensor] = None,
        attention_mask_3: Optional[torch.Tensor] = None,
    ) -> None:
        self._partitions = compute_sentence_partitions(
            attention_mask=attention_mask,
            attention_mask_2=attention_mask_2,
            attention_mask_3=attention_mask_3,
        )
        self._base_device = attention_mask.device
        self._batch_size = attention_mask.size(0)

    def indices(self, kind: str) -> torch.Tensor:
        if kind not in self._KIND_TO_KEY:
            raise ValueError(f"Unknown partition kind: {kind}")
        return self._partitions[self._KIND_TO_KEY[kind]]


def slice_input_batch(
    batch: Dict[str, Any],
    indices: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Slice every tensor/list entry in a batch dict with the provided indices.
    Non tensor/list values are carried as-is.
    """
    if indices.numel() == 0:
        return {}

    sliced: Dict[str, Any] = {}
    idx_list = indices.detach().cpu().tolist()
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            idx = indices.to(value.device)
            subset = value.index_select(0, idx)
            sliced[key] = subset.to(device) if device is not None else subset
        elif isinstance(value, list):
            sliced[key] = [value[i] for i in idx_list]
        else:
            sliced[key] = value
    return sliced


def merge_indexed_outputs(
    batch_size: int,
    device: torch.device,
    *indexed_outputs: Tuple[torch.Tensor, Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Reconstruct a full-batch output dict from multiple sub-batch outputs.
    Each entry is a tuple of (indices, output_dict).
    """
    merged: Dict[str, torch.Tensor] = {}
    for indices, outputs in indexed_outputs:
        if not outputs or indices.numel() == 0:
            continue
        for key, tensor in outputs.items():
            if key not in merged:
                fill_value = float("nan") if tensor.is_floating_point() else 0
                merged[key] = torch.full(
                    (batch_size, *tensor.shape[1:]),
                    fill_value,
                    dtype=tensor.dtype,
                    device=device,
                )
            target = merged[key]
            idx = indices.to(target.device)
            target.index_copy_(0, idx, tensor.to(target.device))
    return merged


__all__ = [
    "extract_unique_strings",
    "flatten_strings",
    "BatchPartitioner",
    "compute_sentence_partitions",
    "slice_input_batch",
    "merge_indexed_outputs",
]

