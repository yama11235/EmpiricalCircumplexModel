import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F

from transformers import PreTrainedModel, AutoModel
import logging
from .modeling_config import (
    build_classifiers,
    load_classifiers
)
from .nGPT_model import justnorm, _is_ngpt_block, _normalize_single_ngpt_block
import os
from .classifier_strategies import (
    _DefaultClassifierStrategy,
)
from .pooler import Pooler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

logger = logging.getLogger(__name__)


class BiEncoderForClassification(PreTrainedModel):
    '''Encoder model with backbone and classification head.'''
    _BACKBONE_ARG_NAMES = (
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "position_ids",
        "head_mask",
        "inputs_embeds",
    )
    def __init__(self, model_config, classifier_configs):
        """
        Args:
            model_config: Model configuration.
            classifier_configs: Classifier configurations. {classifier_name: classifier_config_dict}
        """
        super().__init__(model_config)
        self.config = model_config
        self.model_path = (
            getattr(model_config, "model_name_or_path", None)
            or getattr(model_config, "name_or_path", None)
        )
        self.classifier_configs = classifier_configs
        
        try:
            self.backbone = AutoModel.from_pretrained(
                self.model_path,
                # output_loading_info=True,
                device_map=getattr(model_config, "device_map", "cuda"),
                config=model_config,
                cache_dir=getattr(model_config, "cache_dir", None),
                revision=getattr(model_config, "model_revision", None),
                use_auth_token=True if getattr(model_config, "use_auth_token", None) else None,
                attn_implementation=getattr(model_config, "attn_implementation", "eager"),
                add_pooling_layer=False,
            ).base_model
        except Exception as e:
            self.backbone = AutoModel.from_pretrained(
                self.model_path,
                # output_loading_info=True,
                device_map=getattr(model_config, "device_map", "cuda"),
                config=model_config,
                cache_dir=getattr(model_config, "cache_dir", None),
                revision=getattr(model_config, "model_revision", None),
                use_auth_token=True if getattr(model_config, "use_auth_token", None) else None,
                attn_implementation=getattr(model_config, "attn_implementation", "eager"),
                trust_remote_code=True,
            ).base_model
            
            
        self.output_hidden_states = True
        self.cls_pooler = Pooler(pooler_type="cls")
        self.avg_pooler = Pooler(pooler_type="avg")
        self.max_pooler = Pooler(pooler_type="max")
        self.last_pooler = Pooler(pooler_type="last")
        self.embedding_classifiers = nn.ModuleDict()
        self.clf_configs = {}
        self.classifier_strategy = _DefaultClassifierStrategy()

        if self.classifier_configs:
            # Build classifiers via factory
            self.embedding_classifiers, self.clf_configs = build_classifiers(
                self.classifier_configs, model_config
            )
            self.classifier_save_directory = getattr(
                model_config, 'classifier_save_directory', None
            )
           
        self.post_init()
        # Move classifier to the same device/dtype as backbone
        # print(f"dtype: {next(self.backbone.parameters()).dtype}, device: {next(self.backbone.parameters()).device}")
        self.backbone_device = next(self.backbone.parameters()).device
        for name, classifier in self.embedding_classifiers.items():
            classifier.to(device=self.backbone_device)
            classifier.to(dtype=next(self.backbone.parameters()).dtype)
            
        # --- Detect nGPT-style Block and initial normalization ------------------
        self.use_ngpt_blocks = self._detect_ngpt_blocks()
        if self.use_ngpt_blocks:
            logger.info("Detected nGPT-style classifier block(s); applying initial weight normalization.")
            self.normalize_ngpt_matrices()

    @staticmethod
    def _split_by_batch(tensor: torch.Tensor, batch_sizes: list[int]) -> list[torch.Tensor]:
        splits = []
        offset = 0
        for size in batch_sizes:
            splits.append(tensor[offset : offset + size])
            offset += size
        return splits

    def _pool_and_split(
        self,
        attention_mask: torch.Tensor,
        outputs,
        batch_sizes: list[int],
        target_layer: int | None = None,
    ) -> list[torch.Tensor]:
        layer = target_layer if target_layer is not None else -1
        pooled = self.pooler(attention_mask, outputs, layer)
        return self._split_by_batch(pooled, batch_sizes)

    def _get_sequence_features(
        self,
        outputs,
        batch_sizes: list[int],
        target_layer: int | None = None,
    ) -> list[torch.Tensor]:
        if outputs.hidden_states is None:
            raise ValueError("Hidden states are required for GPT/nGPT classifiers but were not returned by the backbone.")
        layer = target_layer if target_layer is not None else -1
        hidden = outputs.hidden_states[layer]
        return self._split_by_batch(hidden, batch_sizes)

    def _infer_batch_size(self, sentence: dict) -> int:
        for name in self._BACKBONE_ARG_NAMES:
            tensor = sentence.get(name)
            if tensor is not None:
                return tensor.shape[0]
        raise ValueError("Cannot infer batch size from empty sentence inputs.")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """Forward pass for single sentence input only."""
        return self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

    def encode(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        
        ):
        """
        Returns the embedding for single sentence.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=self.output_hidden_states,
        )

        results = {
        # "original_avg": self.avg_pooler(attention_mask, outputs),
        # "original_cls": self.cls_pooler(attention_mask, outputs),
        # "original_max": self.max_pooler(attention_mask, outputs), 
        # "original_last": self.last_pooler(attention_mask, outputs),
    }
        for name, classifier in self.embedding_classifiers.items():
            target_layer = int(self.classifier_configs[name]["layer"])
            sequence_features = self._get_sequence_features(outputs, [input_ids.size(0)], target_layer)[0]
            features = [(sequence_features, attention_mask)]
            results[name] = self.classifier_strategy.single(name, classifier, features)
            
        return results

    def save_pretrained(self, model_save_directory, **kwargs):
        import json
        os.makedirs(model_save_directory, exist_ok=True)
        classifier_save_directory = self.classifier_save_directory if self.classifier_save_directory else model_save_directory
        os.makedirs(classifier_save_directory, exist_ok=True)
        
        # Update config to point to the saved directory
        self.config.classifier_save_directory = os.path.abspath(classifier_save_directory)
        
        # Save Backbone
        if not self.config.freeze_encoder:
            super().save_pretrained(model_save_directory, **kwargs)
        self.config.save_pretrained(model_save_directory)
                
        # Save each classifier
        for name, module in self.embedding_classifiers.items():
            dim_value = self.classifier_configs[name].get('output_dim', self.config.hidden_size)
            param_str = f"{self.classifier_configs[name]['type']}_layer:{self.classifier_configs[name]['layer']}_dim:{dim_value}"
            save_path = os.path.join(classifier_save_directory, param_str,f"{name}_classifier.bin")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(module.state_dict(), save_path)
            self.clf_configs[name].save_pretrained(classifier_save_directory, name)
        
        # Save entire classifier_configs
        classifier_configs_path = os.path.join(model_save_directory, "classifier_configs.json")
        with open(classifier_configs_path, 'w') as f:
            json.dump(self.classifier_configs, f, indent=4)
        
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        import json
        from transformers import AutoConfig
        
        # Load model config
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        
        # Load classifier configs
        classifier_configs_path = os.path.join(pretrained_model_name_or_path, "classifier_configs.json")
        if os.path.exists(classifier_configs_path):
            with open(classifier_configs_path, 'r') as f:
                classifier_configs = json.load(f)
        else:
            classifier_configs = {}
        
        # Get classifier save directory from config
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        classifier_save_directory = saved_config.get('classifier_save_directory', pretrained_model_name_or_path)
                    
        # Initialize model
        model = cls(model_config, classifier_configs)
        
        # Load classifier weights
        if classifier_configs:
            classifier_paths = []
            for name in classifier_configs.keys():
                clf_type = classifier_configs[name]['type']
                clf_layer = classifier_configs[name]['layer']
                dim_value = classifier_configs[name].get('output_dim', model_config.hidden_size)
                param_str = f"{clf_type}_layer:{clf_layer}_dim:{dim_value}"
                clf_path = os.path.join(classifier_save_directory, param_str, f"{name}_classifier.bin")
                classifier_paths.append(clf_path)
            
            loaded_heads = load_classifiers(classifier_configs, model_config, classifier_paths, [])
            model.embedding_classifiers = loaded_heads
        
        # Load backbone weights
        try:
            from safetensors.torch import load_file
            safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                weights = load_file(safetensors_path, device="cpu")
                model.load_state_dict(weights, strict=False)
                logger.info(f"Loaded weights from {safetensors_path}")
        except:
            pytorch_model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                weights = torch.load(pytorch_model_path, map_location="cpu")
                model.load_state_dict(weights, strict=False)
                logger.info(f"Loaded weights from {pytorch_model_path}")

        return model


    def _detect_ngpt_blocks(self) -> bool:
        """
        True if there is a Block with use_nGPT=1 somewhere in embedding_classifiers.
        Do not touch backbone (BERT itself).
        """
        for _, clf in self.embedding_classifiers.items():
            for module in clf.modules():
                if _is_ngpt_block(module):
                    return True
        return False

    def normalize_ngpt_matrices(self) -> None:
        """
        L2 normalize parameters of classifier Block with use_nGPT=1 using nGPT method.
        - Once at initialization
        - Called from train.py every optimizer.step()
        """
        if not getattr(self, "use_ngpt_blocks", False):
            return
        for _, clf in self.embedding_classifiers.items():
            for module in clf.modules():
                if _is_ngpt_block(module):
                    _normalize_single_ngpt_block(module)