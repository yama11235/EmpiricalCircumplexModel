"""Data preprocessing utilities - single sentence only."""
import ast
import argparse
from typing import Any, Dict, List


def parse_dict(dict_str):
    try:
        return ast.literal_eval(dict_str)
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid dictionary literal: {e}")


def get_preprocessing_function(
    tokenizer,
    sentence1_key,
    aspect_key,
    padding,
    max_seq_length,
    model_args,
    classifier_configs=None,
    **kwargs,
):
    """Get preprocessing function for single sentence input."""
    if model_args.encoding_type != 'bi_encoder':
        raise ValueError(f'Invalid model type: {model_args.encoding_type}')

    # Build mapping from dataset column to classifier head name
    column_to_head = {}
    if classifier_configs:
        for head_name, cfg in classifier_configs.items():
            dataset_col = cfg.get("dataset_column", head_name)
            column_to_head[dataset_col] = head_name

    def preprocess_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        is_batch = isinstance(examples[sentence1_key], list)
        if not is_batch:
            examples = {k: [v] for k, v in examples.items()}

        batch_size = len(examples[sentence1_key])
        
        # Tokenize only sentence1
        tokenized = tokenizer(
            examples[sentence1_key],
            padding=padding,
            max_length=max_seq_length,
            truncation=True,
        )
        
        out = dict(tokenized)
        
        # Extract labels and active heads
        active_heads, labels = [], []
        for i in range(batch_size):
            found = False
            for head in aspect_key:
                col = examples.get(head)
                value = None if col is None else (col[i] if isinstance(col, list) else col)
                if value is not None:
                    # Map dataset column to classifier head name
                    head_name = column_to_head.get(head, head)
                    active_heads.append(head_name)
                    labels.append(value)
                    found = True
                    break
            if not found:
                active_heads.append(None)
                labels.append(None)
        
        out["active_heads"] = active_heads
        out["labels"] = labels

        if not is_batch:
            return {k: v[0] if isinstance(v, list) else v for k, v in out.items()}

        return out

    return preprocess_function

