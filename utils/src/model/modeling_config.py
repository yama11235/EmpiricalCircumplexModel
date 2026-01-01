import json
import os
import torch.nn as nn
import torch
from .modeling_classifier import (
    GPTClassifier,
    nGPTClassifier,
)
from typing import Dict, List, Optional

class GPTBlockClassifierConfig:
    def __init__(
        self,
        backbone_dim: int,
        input_dim: int,
        num_heads: int,
        pooler_type: str,
        layer: int = None,
        base_scale: Optional[float] = None,
        bias: bool = False,
        use_ngpt: bool = False,
        dropout: float = 0.1,
        seed: int = 42,
        meta: Optional[dict] = None,
    ):
        if num_heads is None:
            raise ValueError("num_heads must be specified for GPT/nGPT classifiers.")
        self.backbone_dim = backbone_dim
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.pooler_type = pooler_type
        self.layer = layer
        self.base_scale = base_scale if base_scale is not None else (input_dim ** -0.5)
        self.bias = bias
        self.use_ngpt = use_ngpt
        self.dropout = dropout  
        self.seed = seed
        self.meta = meta or {}

    def to_dict(self):
        return {
            'input_dim': self.input_dim,
            'num_heads': self.num_heads,
            'pooler_type': self.pooler_type,
            'layer': self.layer,
            'base_scale': self.base_scale,
            'bias': self.bias,
            'use_ngpt': self.use_ngpt,
            'meta': self.meta,
        }

    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            input_dim=config_dict['input_dim'],
            num_heads=config_dict['num_heads'],
            pooler_type=config_dict['pooler_type'],
            layer=config_dict['layer'],
            base_scale=config_dict.get('base_scale'),
            bias=config_dict.get('bias', False),
            use_ngpt=config_dict.get('use_ngpt', False),
            meta=config_dict.get('meta', {}),
        )

    def save_pretrained(self, save_path: str, classifier_name: str):
        config_dict = self.to_dict()
        config_dict['classifier_name'] = classifier_name
        block_type = "ngpt_block" if self.use_ngpt else "gpt_block"
        save_path = os.path.join(
            save_path,
            f"{block_type}:{self.layer}_dim:{self.input_dim}",
            f"{classifier_name}.json",
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

def build_classifiers(classifier_configs: dict, model_config) -> (nn.ModuleDict, dict):
    """
    Factory to build classifier modules and their configs.

    Args:
        classifier_configs: Mapping from classifier name to its parameter dict.
        model_config: Model configuration with attributes hidden_size, num_hidden_layers, etc.
    Returns:
        modules: nn.ModuleDict of classifier modules.
        clf_configs: Dict of classifier configuration objects.
    """
    modules = nn.ModuleDict()
    clf_configs = {}

    for name, params in classifier_configs.items():
        params = params.copy()
        params.setdefault("name", name)
        ctype = params.get("type")
        if ctype is None:
            raise ValueError(f"Classifier {name} - 'type' is required in the config.")
        ctype_key = ctype.lower() if isinstance(ctype, str) else ctype

        input_dim = params.get("input_dim", model_config.hidden_size)
        hidden2head = {2:1, 4:1, 8:2, 16:2, 32:2, 64:4, 128:8, 256:8, 512:8, 768:12, 1024:16, 2560:32, 3072:24, 4096:32}
        num_heads = params.get("num_heads", hidden2head[input_dim] if input_dim in hidden2head else getattr(model_config, "num_attention_heads", None))
        if num_heads is None:
            raise ValueError(f"Classifier {name} requires 'num_heads' in params or model_config.num_attention_heads.")
        base_scale = params.get(
            "base_scale",
            1.0 / (model_config.hidden_size ** 0.5),
        )
        cfg = GPTBlockClassifierConfig(
            backbone_dim=model_config.hidden_size,
            input_dim=input_dim,
            num_heads=num_heads,
            pooler_type=params.get("pooler_type", "cls"),
            layer=params.get("layer", -1),
            base_scale=base_scale,
            bias=params.get("bias", False),
            use_ngpt=(ctype_key == 'ngpt'),
            dropout=params.get("dropout", 0.1),
            seed=getattr(model_config, "seed", 42),
            meta={
                "name": name,
                "type": "nGPT" if ctype_key == 'ngpt' else "GPT",
                "objective": params.get("objective", "SINCERE"),
                "distance": params.get("distance", "cosine"),
            },
        )
        print(f"Classifier Config for {name}: {cfg.to_dict()}")
        module_cls = nGPTClassifier if ctype_key == 'ngpt' else GPTClassifier
        module = module_cls(cfg)

        modules[name] = module
        clf_configs[name] = cfg

    return modules, clf_configs

def load_classifiers(classifier_configs: Dict, model_config: Dict, save_dir: List, classifier_freeze: List) -> nn.ModuleDict:
    """
    Builds classifiers and loads their weights from files.

    Args:
        classifier_configs: Mapping from classifier name to its parameter dict.
        model_config: Model configuration.
        save_dir: List of Directory where classifier weights are stored.

    Returns:
        ModuleDict of loaded classifier modules.
    """
    modules, _ = build_classifiers(classifier_configs, model_config)

    for name, classifier_path in zip(list(modules.keys()), save_dir):
        weight_path = classifier_path
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Classifier weight file for {name} not found at {weight_path}")
        state = {} if not os.path.getsize(weight_path) else torch.load(weight_path, map_location="cpu")
        modules[name].load_state_dict(state)
        
        if name in classifier_freeze:
            for param in modules[name].parameters():
                param.requires_grad = False
            print(f"Classifier {name} is frozen and will not be trained.")

    return modules
