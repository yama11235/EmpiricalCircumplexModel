"""Model subpackage that hosts encoder architectures and classifier utilities."""

from __future__ import annotations

from .pooler import Pooler
from .classifier_strategies import (
    _ClassifierStrategy,
    _DefaultClassifierStrategy,
)

__all__ = [
    "Pooler",
    "_ClassifierStrategy",
    "_DefaultClassifierStrategy",
]
