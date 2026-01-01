"""Model utilities."""
from .modeling_encoders import BiEncoderForClassification
import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from transformers import PreTrainedTokenizerBase

@dataclass
class DataCollatorForBiEncoder:
    """Data collator for single-sentence input only."""
    tokenizer: PreTrainedTokenizerBase
    padding: Optional[bool | str] = "max_length"
    pad_to_multiple_of: Optional[int] = None
    dtype: torch.dtype = torch.float32

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        active_heads = [f["active_heads"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Extract only sentence1 fields
        batch_features = []
        for f in features:
            batch_features.append({
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"],
                **({"token_type_ids": f["token_type_ids"]} if "token_type_ids" in f else {})
            })
        
        # Pad batch
        batch = self.tokenizer.pad(
            batch_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        batch["labels"] = torch.tensor(labels, dtype=self.dtype)
        batch["active_heads"] = active_heads
        return batch

def get_model(model_args):
    if model_args.encoding_type == 'bi_encoder':
        return BiEncoderForClassification
    raise ValueError(f'Invalid model type: {model_args.encoding_type}')
