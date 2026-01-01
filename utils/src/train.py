"""
Adapted code from HuggingFace run_glue.py

Author: Ameet Deshpande, Carlos E. Jimenez

Main training script for multi-classifier embedding models.
"""
import logging
import os
import sys
import random
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __package__ is None or __package__ == "":
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    grandparent_dir = parent_dir.parent
    sys.path.insert(0, str(grandparent_dir))
    __package__ = "utils.src"

import datasets
import transformers
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint

from utils.src.config import TrainingArguments, DataTrainingArguments, ModelArguments
from utils.src.data import load_raw_datasets, prepare_label_mappings
from utils.src.training import (
    setup_model_and_config,
    setup_tokenizer,
    prepare_datasets,
    create_trainer,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]),
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        filename="logs/train.log",
        filemode="a",
    )
    training_args.log_level = "info"
    training_args.remove_unused_columns = False
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info("Training/evaluation parameters %s" % training_args)
    
    # Check for checkpoints
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.warning(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    # Set seed
    seed = training_args.seed
    random.seed(seed)
    
    # Load datasets
    raw_datasets = load_raw_datasets(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        seed=seed,
    )
    
    # Prepare label mappings
    (
        raw_datasets,
        labels,
        id2label,
        label2id,
        aspect_key,
        classifier_configs,
        classifier_configs_for_trainer,
        label_name_mappings,
    ) = prepare_label_mappings(
        raw_datasets=raw_datasets,
        model_args=model_args,
        data_args=data_args,
    )
    
    print("Classifier Configs:", classifier_configs)
    print("Classifier Configs for Trainer:", classifier_configs_for_trainer)
    
    # Setup model and config
    config, model, use_ngpt_riemann = setup_model_and_config(
        model_args=model_args,
        training_args=training_args,
        labels=list(classifier_configs_for_trainer.keys()),
        id2label=id2label,
        label2id=label2id,
        classifier_configs=classifier_configs,
    )
    
    # Setup tokenizer
    tokenizer = setup_tokenizer(model_args)
    
    # Prepare datasets (single sentence only)
    train_dataset, eval_dataset, predict_dataset, max_train_samples = prepare_datasets(
        raw_datasets=raw_datasets,
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        aspect_key=aspect_key,
        classifier_configs=classifier_configs_for_trainer,
    )
    
    # Create trainer
    all_heads = list(classifier_configs_for_trainer.keys())
    dataset_columns = set(cfg.get("dataset_column") for cfg in classifier_configs_for_trainer.values() if cfg.get("dataset_column"))
    for col in dataset_columns:
        if col not in all_heads:
            all_heads.append(col)
    id2_head = {i: head for i, head in enumerate(all_heads)}
    print("ID to Head Mapping:", id2_head)
    print("dataset_columns:", dataset_columns)
    print("label_name_mappings:", label_name_mappings)

    trainer, trainer_state = create_trainer(
        model=model,
        config=config,
        training_args=training_args,
        classifier_configs_for_trainer=classifier_configs_for_trainer,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        label_name_mappings=label_name_mappings,
        use_ngpt_riemann=use_ngpt_riemann,
        id2_head=id2_head,
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
                    
        trainer.save_model()
        trainer.save_state()
    

if __name__ == "__main__":
    main()
