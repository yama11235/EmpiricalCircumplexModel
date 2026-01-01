"""Custom trainer with multi-head classifier support - simplified version."""
import os
import numpy as np
import torch
from datasets import Dataset
from transformers import Trainer
from typing import List, Dict, Any, Optional
from collections import defaultdict

from utils.src.data.batch_utils import (
    extract_unique_strings,
    flatten_strings,
)
from utils.src.training.objectives import (
    CircularCSEObjective,
    HeadObjective,
    SINCEREObjective,
    SoftCSEObjective,
)
from utils.src.loss_function import (
    compute_single_loss,
)

class CustomTrainer(Trainer):
    """Custom trainer for multi-head classifier training."""
    
    def __init__(
        self,
        *args,
        classifier_configs,
        dtype=torch.float16,
        label_name_mappings: Optional[Dict[str, Dict[int, str]]] = None,
        id2_head: Optional[Dict[int, str]] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.classifier_configs = classifier_configs
        self.dtype = dtype
        self.label_name_mappings: Dict[str, Dict[int, str]] = label_name_mappings or {}

        self.loss_fns = {}
        self.sincere_params: Dict[str, Dict[str, Any]] = {}
        self.circular_cse_params: Dict[str, Dict[str, Any]] = {}
        self.soft_cse_params: Dict[str, Dict[str, Any]] = {}
        self.head_to_column: Dict[str, str] = {}
        self.column_to_heads: Dict[str, List[str]] = {}  # Map dataset columns to classifier heads

        for name, cfg in classifier_configs.items():
            dataset_col = cfg.get("dataset_column", name)
            self.head_to_column[name] = dataset_col
            # Build reverse mapping: column -> [heads]
            if dataset_col not in self.column_to_heads:
                self.column_to_heads[dataset_col] = []
            self.column_to_heads[dataset_col].append(name)
            
            obj = cfg["objective"]
            
            if obj == "CircularCSE":
                id2label = cfg.get("id2label", {})
                dataset_column = cfg.get("dataset_column", name)
                if not id2label and dataset_column in self.label_name_mappings:
                    id2label = self.label_name_mappings[dataset_column]
                self.circular_cse_params[name] = {
                    "angle_map": cfg.get("angle_map", {}),
                    "id2label": id2label,
                    "inbatch_pairs": cfg.get("inbatch_pairs", 256),
                }
            elif obj == "SoftCSE":
                id2label = cfg.get("id2label", {})
                dataset_column = cfg.get("dataset_column", name)
                if not id2label and dataset_column in self.label_name_mappings:
                    id2label = self.label_name_mappings[dataset_column]
                self.soft_cse_params[name] = {
                    "tau": cfg.get("tau", 1.0),
                    "pos_pairs": cfg.get("inbatch_positive_pairs", 10),
                    "neg_pairs": cfg.get("inbatch_negative_pairs", 64),
                    "use_angle_map": cfg.get("use_angle_map", True),
                    "similarity_calculator": cfg.get("similarity_calculator", None),
                    "angle_map": cfg.get("angle_map", {}),
                    "id2label": id2label,
                }
            elif obj == "SINCERE":
                self.sincere_params[name] = {
                    "tau": cfg.get("tau", 1.0),
                    "pos_pairs": cfg.get("inbatch_positive_pairs", 10),
                    "neg_pairs": cfg.get("inbatch_negative_pairs", 64),
                }
        
        if id2_head is not None:
            self.head2idx = {head: i for i, head in id2_head.items()}
            self.idx2head = id2_head
        else:
            all_heads = list(classifier_configs.keys())
            dataset_columns = set(cfg.get("dataset_column") for cfg in classifier_configs.values() if cfg.get("dataset_column"))
            for col in dataset_columns:
                if col not in all_heads:
                    all_heads.append(col)
            self.head2idx = {head: i for i, head in enumerate(all_heads)}
            self.idx2head = {idx: head for head, idx in self.head2idx.items()}
        
        self.head_objectives: Dict[str, HeadObjective] = {}
        for name, cfg in classifier_configs.items():
            obj = cfg.get("objective")
            # Create a config copy and inject id2label from label_name_mappings if needed
            cfg_with_id2label = dict(cfg)
            
            if obj in ["CircularCSE", "SoftCSE"]:
                # If angle_map is present but id2label is not, we need to create id2label
                if "angle_map" in cfg_with_id2label and cfg_with_id2label["angle_map"]:
                    if "id2label" not in cfg_with_id2label or not cfg_with_id2label["id2label"]:
                        # Try to get from label_name_mappings using dataset_column
                        dataset_column = cfg_with_id2label.get("dataset_column", name)
                        if dataset_column in self.label_name_mappings:
                            cfg_with_id2label["id2label"] = self.label_name_mappings[dataset_column]
                        else:
                            raise ValueError(
                                f"id2label not found for head '{name}' with dataset_column '{dataset_column}'. "
                                f"Please ensure label_name_mappings is correctly set up."
                            )
                
                if obj == "CircularCSE":
                    self.head_objectives[name] = CircularCSEObjective(name, cfg_with_id2label)
                else:  # SoftCSE
                    self.head_objectives[name] = SoftCSEObjective(name, cfg_with_id2label)
            elif obj == "SINCERE":
                self.head_objectives[name] = SINCEREObjective(name, cfg_with_id2label)
            else:
                raise ValueError(f"Unknown objective type: {obj}. Supported: SINCERE, CircularCSE, SoftCSE")
        
        print(f"Head Objectives: {self.head_objectives}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss computation using objectives."""
        
        device = next(model.parameters()).device
        active_heads = extract_unique_strings(inputs["active_heads"])
        
        # Store current inputs for similarity calculator access
        self._current_inputs = inputs
        
        # Use centralized loss function
        loss, outputs = compute_single_loss(
            trainer=self,
            model=model,
            inputs=inputs,
            active_heads=active_heads,
            device=device,
        )
        
        if loss is None:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        if not return_outputs:
            return loss
        return loss, outputs

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            loss, logits = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss.detach() if loss is not None else loss, None, None)

        labels_tensor = inputs["labels"]
        ah_list = flatten_strings(inputs["active_heads"])
        ah_ids = [self.head2idx[h] for h in ah_list]
        ah_tensor = torch.tensor(ah_ids, device=loss.device, dtype=torch.long)

        label_dict = {"labels": labels_tensor, "active_heads": ah_tensor}
        head_name = self.idx2head[ah_ids[0]]  # Assuming single head per batch for prediction
        return loss.detach() if loss is not None else loss, logits[head_name], label_dict

