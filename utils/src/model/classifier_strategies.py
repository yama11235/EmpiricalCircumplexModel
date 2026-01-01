"""Strategies and helpers for classifier heads."""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:  # pragma: no cover
    from modeling_encoders import BiEncoderForClassification

__all__ = [
    "_ClassifierStrategy",
    "_DefaultClassifierStrategy",
]


class _ClassifierStrategy:
    def single(
        self,
        name: str,
        classifier: nn.Module,
        features: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {}


class _DefaultClassifierStrategy(_ClassifierStrategy):
    def single(self, name, classifier, features):
        seq, mask = features[0]
        embedding = classifier.encode(seq, mask).to(seq.dtype)
        return embedding
