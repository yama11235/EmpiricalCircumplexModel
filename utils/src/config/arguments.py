"""Argument definitions for training."""
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class TrainingArguments(HFTrainingArguments):
    """Extended training arguments with custom fields."""
    
    log_time_interval: int = field(
        default=15,
        metadata={
            "help": (
                "Log at each `log_time_interval` seconds. "
                "Default will be to log every 15 seconds."
            )
        },
    )
    seed: Optional[int] = field(
        default=42,
        metadata={
            "help": "A seed for reproducible training. If not set, will use a random seed."
        },
    )


@dataclass
class DataTrainingArguments:
    """Arguments for data loading and preprocessing."""

    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the cached training and evaluation sets."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[List[str]] = field(
        default=None,
        metadata={"help": "A list of csv or json files containing the training data."},
    )
    validation_file: Optional[List[str]] = field(
        default=None,
        metadata={"help": "A list of csv or json files containing the validation data."},
    )
    test_file: Optional[List[str]] = field(
        default=None,
        metadata={"help": "A list of csv or json files containing the test data."},
    )

    def __post_init__(self):
        def _get_extension(x) -> str:
            """Get file extension from string or list."""
            if isinstance(x, list):
                if len(x) == 0:
                    raise ValueError("file list must contain at least one path")
                path = x[0]
            else:
                path = x
            return path.split(".")[-1]

        val_ext = _get_extension(self.validation_file)
        
        if self.train_file:
            train_ext = _get_extension(self.train_file)
            assert train_ext in ["csv", "json"], "`train_file` should be csv or json."
            assert train_ext == val_ext, "`train_file` and `validation_file` must have same extension."
        
        if self.test_file:
            test_ext = _get_extension(self.test_file)
            assert test_ext in ["csv", "json"], "`test_file` should be csv or json."
            assert test_ext == val_ext, "`test_file` and `validation_file` must have same extension."


@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    use_flash_attention: str = field(
        default="eager",
        metadata={"help": "Whether to use flash attention."},
    )
    encoding_type: Optional[str] = field(
        default="bi_encoder",
        metadata={
            "help": "What kind of model to choose. Options:\
            1) cross_encoder: Full encoder model.\
            2) bi_encoder: Bi-encoder model.\
            3) tri_encoder: Tri-encoder model."
        },
    )
    freeze_encoder: Optional[bool] = field(
        default=False, metadata={"help": "Freeze encoder weights."}
    )
    classifier_configs: Optional[str] = field(
        default=None,
        metadata={"help": "classifier configs in json format."},
    )
    aspect_key: Optional[str] = field(
        default=None,
        metadata={"help": "Aspect key(s) for multi-aspect classification. Can be a comma-separated string or single string."},
    )
    objective: Optional[str] = field(
        default="SINCERE",
        metadata={"help": "Training objective. Options: SINCERE, CircularCSE, SoftCSE. Default is 'SINCERE'."},
    )
