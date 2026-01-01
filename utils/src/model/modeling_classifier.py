from torch import nn
import torch
from types import SimpleNamespace

from .nGPT_model import Block
from .pooler import Pooler

class _GPTBlockClassifierBase(nn.Module):
    """
    Shared logic for GPT-style transformer block classifiers.

    Takes a pooled embedding, runs it through a single transformer block borrowed
    from the GPT/nGPT implementation, and projects it with a learnable head.
    """

    def __init__(self, config, use_ngpt: bool):
        super().__init__()
        self.config = config
        self.use_ngpt = use_ngpt
        self.dropout = nn.Dropout(config.dropout)
        self.backbone_dim = config.backbone_dim
        self.input_dim = config.input_dim
        self.seed = config.seed

        block_config = SimpleNamespace(
            n_embd=self.input_dim,
            n_head=config.num_heads,
            base_scale=config.base_scale,
            use_nGPT=1 if use_ngpt else 0,
            bias=config.bias,
        )
        # iblock is unused inside Block, so we can fix it at 0.
        self.transformer = Block(block_config, iblock=0)

        pooler_type = config.pooler_type
        self.pooler = Pooler(pooler_type)
        
        # if input_dim < backbone_dim, we randomly select a subset of dimensions and fix them
        if self.input_dim < self.backbone_dim:
            torch.manual_seed(self.seed)
            self.register_buffer("selected_indices", torch.randperm(self.backbone_dim)[:self.input_dim])

    def _ensure_sequence_dim(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.dim() == 2:
            return embedding.unsqueeze(1)
        if embedding.dim() == 3:
            return embedding
        raise ValueError(
            f"Expected 2D or 3D tensor for GPT-style classifier, got shape {embedding.shape}"
        )

    def forward(self, embedding: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedding = self.dropout(embedding)
        sequence = self._ensure_sequence_dim(embedding)
        
        if self.input_dim < sequence.size(-1):
            # truncate the embedding to selected indices
            sequence = sequence[:, :, self.selected_indices]
            
        transformed = self.transformer(sequence, attention_mask)
        return self.pooler(attention_mask, transformed)

    def encode(self, embedding: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.forward(embedding, attention_mask)


class GPTClassifier(_GPTBlockClassifierBase):
    """Classifier that uses a standard GPT transformer block."""

    def __init__(self, config):
        super().__init__(config, use_ngpt=False)


class nGPTClassifier(_GPTBlockClassifierBase):
    """Classifier that uses the custom nGPT transformer block."""

    def __init__(self, config):
        super().__init__(config, use_ngpt=True)
    
