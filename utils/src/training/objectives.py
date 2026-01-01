"""Head-specific objective functions for multi-task learning."""
from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from utils.src.training.sincere_loss import compute_sincere_loss
from utils.src.training.circular_cse_loss import compute_circular_cse_loss
from utils.src.training.soft_cse_loss import compute_softcse_loss


class HeadObjective(ABC):
    """Base class providing a uniform interface for head-specific losses."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name = name
        self.config = config

    # ---- Single sentence objectives -------------------------------------------------
    def compute_single(
        self,
        trainer: Any,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return None


class SINCEREObjective(HeadObjective):
    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__(name, config)
        self.tau = float(config.get("tau", 1.0))
        self.log_of_sum = bool(config.get("log_of_sum", False))

    def compute_single(
        self,
        trainer: Any,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if embeddings.numel() == 0:
            return None
        tau = max(self.tau, 1e-6)
        return compute_sincere_loss(
            embeddings=embeddings,
            labels=labels.long(),
            tau=tau,
            log_of_sum=self.log_of_sum,
        )


class CircularCSEObjective(HeadObjective):
    """CircularCSE: angle-based contrastive learning objective."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__(name, config)
        self.angle_map = config.get("angle_map", {})
        self.id2label = config.get("id2label", {})
        self.margin = float(config.get("margin", 0.0))

    def compute_single(
        self,
        trainer: Any,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if embeddings.numel() == 0:
            return None
        if not self.angle_map:
            return None
        
        # Use id2label from config if available, otherwise let compute_circular_cse_loss infer it
        id2label_to_use = self.id2label if self.id2label else None
        
        return compute_circular_cse_loss(
            embeddings=embeddings,
            labels=labels.long(),
            angle_map=self.angle_map,
            id2label=id2label_to_use,
            margin=self.margin,
        )


class SoftCSEObjective(HeadObjective):
    """SoftCSE: weighted contrastive learning objective with teacher similarity."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__(name, config)
        self.tau = float(config.get("tau", 1.0))
        self.use_angle_map = config.get("use_angle_map", True)  # Default to True for backward compatibility
        self.angle_map = config.get("angle_map", {})
        self.id2label = config.get("id2label", {})
        self.log_of_sum = bool(config.get("log_of_sum", False))
        
        # Initialize similarity calculator model if specified
        self.similarity_calculator = None
        self.similarity_tokenizer = None
        similarity_model_name = config.get("similarity_calculator", None)
        
        if similarity_model_name:
            print(f"Loading similarity calculator model: {similarity_model_name}")
            self.similarity_tokenizer = AutoTokenizer.from_pretrained(similarity_model_name)
            self.similarity_calculator = AutoModel.from_pretrained(similarity_model_name)
            
            # Freeze the similarity calculator model
            self.similarity_calculator.eval()
            for param in self.similarity_calculator.parameters():
                param.requires_grad = False
            
            print(f"Similarity calculator loaded and frozen: {similarity_model_name}")

    def compute_single(
        self,
        trainer: Any,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if embeddings.numel() == 0:
            return None
        
        # Compute pairwise similarity based on use_angle_map flag
        pairwise_sim = None
        if self.use_angle_map and self.angle_map:
            # Use angle-based similarity when use_angle_map=True and angle_map exists
            pairwise_sim = self._compute_angle_based_similarity(labels, embeddings.device)
        elif not self.use_angle_map and self.similarity_calculator is not None:
            # Use similarity calculator model when use_angle_map=False
            pairwise_sim = self._compute_similarity_with_calculator(trainer, embeddings.device)
        # If use_angle_map=False and no similarity_calculator, pairwise_sim remains None
        # and compute_softcse_loss will use cosine similarity normalized to [0,1]
        
        tau = max(self.tau, 1e-6)
        return compute_softcse_loss(
            embeddings=embeddings,
            labels=labels.long(),
            tau=tau,
            pairwise_sim=pairwise_sim,
            log_of_sum=self.log_of_sum,
        )
    
    def _compute_angle_based_similarity(
        self,
        labels: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute pairwise similarity matrix based on angle differences using standard Cosine scaling.
        
        Args:
            labels: Label IDs [B]
            device: Device to place the similarity matrix
            
        Returns:
            Pairwise similarity matrix [B, B] where:
            - angle_diff = 0° → similarity = 1.0
            - angle_diff = 180° → similarity = 0.0
            Using (cos(theta) + 1) / 2 mapping.
        """
        B = labels.size(0)
        
        # Build label_to_angle mapping
        label_to_angle = {}
        for angle_str, label_name in self.angle_map.items():
            angle_val = float(angle_str)
            label_to_angle[label_name] = angle_val
        
        # Use id2label from config
        if not self.id2label:
            raise ValueError(
                f"id2label must be provided for SoftCSE with angle_map. "
                f"Please ensure label_name_mappings is correctly set up."
            )
        
        # Convert label IDs to angles
        angles = []
        for lbl in labels:
            label_id = int(lbl.item())
            label_name = self.id2label.get(label_id)
            if label_name and label_name in label_to_angle:
                angles.append(label_to_angle[label_name])
            else:
                angles.append(0.0)
        
        angles_tensor = torch.tensor(angles, device=device, dtype=torch.float32)
        
        # Compute pairwise angle differences [B, B]
        angle_diff = torch.abs(angles_tensor.unsqueeze(0) - angles_tensor.unsqueeze(1))
        
        # Handle circular nature: min(diff, 360 - diff) -> Range [0, 180]
        angle_diff = torch.min(angle_diff, 360.0 - angle_diff)
        
        # --- Modified Section: Standard Cosine Mapping ---
        
        # 1. Convert degrees to radians: [0, 180] -> [0, π]
        #    Using torch.pi directly
        angle_rad = angle_diff * (torch.pi / 180.0)
        
        # 2. Apply Cosine transformation
        #    cos(0) = 1.0
        #    cos(π) = -1.0
        cos_sim = torch.cos(angle_rad)
        
        return cos_sim
    
    def _compute_similarity_with_calculator(
        self,
        trainer: Any,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute pairwise similarity matrix using the similarity calculator model.
        
        Args:
            trainer: Trainer object containing input data
            device: Device to place the similarity matrix
            
        Returns:
            Pairwise similarity matrix [B, B] computed by similarity_calculator
        """
        if self.similarity_calculator is None or self.similarity_tokenizer is None:
            raise ValueError("similarity_calculator model not initialized")
        
        # Get input texts from trainer's current batch
        # Assuming trainer has access to the current inputs
        if not hasattr(trainer, '_current_inputs') or trainer._current_inputs is None:
            raise ValueError("Trainer does not have _current_inputs attribute")
        
        inputs = trainer._current_inputs
        
        """
        inputs = {
            'input_ids': Tensor of shape [B, seq_len],
            'attention_mask': Tensor of shape [B, seq_len],
            'labels': Tensor of shape [B],
            ...}
        """
        
        # Tokenize texts
        with torch.no_grad():
            encoded = {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
            }
            if 'token_type_ids' in inputs:
                encoded['token_type_ids'] = inputs['token_type_ids']

            # Then move to device
            encoded = {k: v.to(device) for k, v in encoded.items()}
            # Move model to device if needed
            if next(self.similarity_calculator.parameters()).device != device:
                self.similarity_calculator.to(device)
            
            # Get embeddings from similarity calculator
            outputs = self.similarity_calculator(**encoded)
            
            # Use mean pooling over sequence
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded['attention_mask']
            
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            calc_embeddings = sum_embeddings / sum_mask
            
            # Normalize embeddings
            calc_embeddings = F.normalize(calc_embeddings, dim=1)
            
            # Compute pairwise cosine similarity [B, B]
            pairwise_sim = calc_embeddings @ calc_embeddings.t()
        
        return pairwise_sim


__all__ = [
    "HeadObjective",
    "SINCEREObjective",
    "CircularCSEObjective",
    "SoftCSEObjective",
]
