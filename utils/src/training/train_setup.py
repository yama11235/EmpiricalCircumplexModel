"""Training setup utilities."""
import os
import logging
import random
from typing import Optional, Dict, Tuple
import torch
from transformers import AutoConfig, AutoTokenizer, PrinterCallback
from utils.src.data.preprocessing import get_preprocessing_function
from utils.src.model.modeling_utils import DataCollatorForBiEncoder, get_model
from utils.src.model.nGPT_model import NGPTWeightNormCallback
from utils.src.clf_trainer import CustomTrainer
from utils.src.progress_logger import LogCallback


logger = logging.getLogger(__name__)


def setup_model_and_config(
    model_args,
    training_args,
    labels,
    id2label,
    label2id,
    classifier_configs,
) -> Tuple[any, any, bool]:
    """
    Setup model configuration and initialize model.
    
    Returns:
        Tuple of (config, model, use_ngpt_riemann)
    """
    # Don't pass id2label and label2id to AutoConfig as they may be inconsistent
    # The model will handle label mappings internally through classifier_configs
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        trust_remote_code=True,
    )
    
    if training_args.fp16:
        config.torch_dtype = torch.float32
    elif training_args.bf16:
        config.torch_dtype = torch.bfloat16
        
    model_cls = get_model(model_args)
    config.update(
        {
            "freeze_encoder": model_args.freeze_encoder,
            "use_auth_token": model_args.use_auth_token,
            "model_revision": model_args.model_revision,
            "cache_dir": model_args.cache_dir,
            "model_name_or_path": model_args.model_name_or_path,
            "attn_implementation": model_args.use_flash_attention,
            "seed": training_args.seed,
        }
    )
    
    model = model_cls(model_config=config, classifier_configs=classifier_configs)
    
    # Freeze or unfreeze encoder
    for param in model.backbone.parameters():
        param.requires_grad = not model_args.freeze_encoder

    # Check for nGPT and adjust optimizer settings
    use_ngpt_riemann = bool(getattr(model, "use_ngpt_blocks", False))
    if use_ngpt_riemann:
        logger.info(
            "nGPT-style classifier detected. Enabling pseudo-Riemann weight normalization "
            "and nGPT-friendly optimizer settings."
        )
        if training_args.weight_decay != 0.0:
            logger.warning(
                f"Overriding weight_decay from {training_args.weight_decay} to 0.0 for nGPT."
            )
            training_args.weight_decay = 0.0

        if getattr(training_args, "warmup_steps", 0) != 0:
            logger.warning(
                f"Overriding warmup_steps from {training_args.warmup_steps} to 0 for nGPT."
            )
            training_args.warmup_steps = 0

        if getattr(training_args, "warmup_ratio", 0.0) != 0.0:
            logger.warning(
                f"Overriding warmup_ratio from {training_args.warmup_ratio} to 0.0 for nGPT."
            )
            training_args.warmup_ratio = 0.0
    else:
        logger.info(
            "No nGPT-style classifier detected. Training will use standard optimizer settings."
        )

    logger.debug("Model architecture: %s", model)
    return config, model, use_ngpt_riemann


def setup_tokenizer(model_args):
    """Setup tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True,
    )
    
    # Set pad_token to eos_token if pad_token is not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"pad_token was not set. Using eos_token as pad_token: {tokenizer.eos_token}")
    
    return tokenizer


def prepare_datasets(
    raw_datasets,
    tokenizer,
    data_args,
    model_args,
    training_args,
    aspect_key,
    classifier_configs=None,
):
    """
    Tokenize and prepare datasets.
    
    Returns:
        Tuple of (train_dataset, eval_dataset, predict_dataset, max_train_samples)
    """
    # Determine padding strategy
    if data_args.pad_to_max_length:
        padding = "longest"
    else:
        padding = False
        
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            "The max_seq_length passed (%d) is larger than the maximum length for the "
            "model (%d). Using max_seq_length=%d.",
            data_args.max_seq_length,
            tokenizer.model_max_length,
            tokenizer.model_max_length,
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    
    # Get preprocessing function (single sentence only)
    preprocess_function = get_preprocessing_function(
        tokenizer,
        sentence1_key="sentence1",
        aspect_key=aspect_key,
        padding=padding,
        max_seq_length=max_seq_length,
        model_args=model_args,
        classifier_configs=classifier_configs,
    )
    batched = False
    
    # Tokenize datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        logger.debug("Raw datasets before tokenization: %s", raw_datasets)
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=batched,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
            remove_columns=raw_datasets["train"].column_names,
        )
    
    # Prepare train dataset
    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        train_dataset_size = len(train_dataset)
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else train_dataset_size
        )
        
        # Log samples
        sample_count = min(3, train_dataset_size)
        indices = random.sample(range(train_dataset_size), sample_count) if sample_count > 0 else []
        for index in indices:
            input_ids = train_dataset[index]["input_ids"]
            logger.info(f"tokens: {tokenizer.decode(input_ids)}")
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    
    # Prepare eval dataset
    eval_dataset = None
    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
    
    # Prepare predict dataset
    predict_dataset = None
    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
    
    return train_dataset, eval_dataset, predict_dataset, max_train_samples

def create_trainer(
    model,
    config,
    training_args,
    classifier_configs_for_trainer,
    tokenizer,
    train_dataset,
    eval_dataset,
    label_name_mappings,
    use_ngpt_riemann,
    id2_head,
):
    """
    Create and configure the trainer.
    
    Returns:
        Tuple of (trainer, trainer_state)
    """
    collator_dtype = getattr(config, "torch_dtype", torch.float32)
    
    logger.debug(
        "Torch dtype: %s, collator dtype: %s", config.torch_dtype, collator_dtype
    )
    
    data_collator = DataCollatorForBiEncoder(
        tokenizer=tokenizer,
        padding="longest",
        pad_to_multiple_of=None,
        dtype=collator_dtype,
    )
    
    trainer_state = {"trainer": None}
    
    ngpt_callback = NGPTWeightNormCallback(enabled=use_ngpt_riemann)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        classifier_configs=classifier_configs_for_trainer,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=None,
        tokenizer=tokenizer,
        callbacks=[LogCallback, ngpt_callback],
        dtype=collator_dtype,
        label_name_mappings=label_name_mappings,
        id2_head=id2_head,
    )
    trainer_state["trainer"] = trainer
    
    trainer.remove_callback(PrinterCallback)
    
    return trainer, trainer_state
