"""Dataset loading utilities."""
import logging
from typing import Dict, List, Tuple
import datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict

logger = logging.getLogger(__name__)


def load_raw_datasets(
    model_args,
    data_args,
    training_args,
    seed: int,
) -> Tuple[DatasetDict, bool]:
    """
    Load raw datasets from files.
    
    Args:
        model_args: Model arguments
        data_args: Data arguments
        training_args: Training arguments
        seed: Random seed for shuffling
        
    Returns:
        DatasetDict
    """
    data_files: Dict[str, List[str]] = {}

    if training_args.do_train and data_args.train_file:
        data_files["train"] = data_args.train_file
        logger.info("Load train files: %s", data_args.train_file)
    if data_args.validation_file:
        data_files["validation"] = data_args.validation_file
        logger.info("Load validation files: %s", data_args.validation_file)
    if data_args.test_file:
        data_files["test"] = data_args.test_file
        logger.info("Load test files: %s", data_args.test_file)
    elif training_args.do_predict:
        raise ValueError("test_file argument is missing. required for do_predict.")

    example_file = next((files[0] for files in data_files.values() if files), None)
    if example_file is None:
        raise ValueError(
            "At least one of `train_file`, `validation_file`, or `test_file` must be provided."
        )

    file_type = "csv" if example_file.endswith(".csv") else "json"
    logger.debug("Resolved file type for datasets: %s", file_type)

    raw_datasets: Dict[str, datasets.Dataset] = {}

    for split in ["train", "validation", "test"]:
        paths = data_files.get(split, [])
        if not paths:
            continue

        if split == "train":
            sample_limit = data_args.max_train_samples
        elif split == "validation":
            sample_limit = data_args.max_eval_samples
        else:
            sample_limit = data_args.max_predict_samples

        first = load_dataset(
            file_type,
            data_files={split: [paths[0]]},
            split=split,
            cache_dir=model_args.cache_dir,
        )

        if sample_limit is not None and len(first) > sample_limit:
            first = first.shuffle(seed=seed).select(range(sample_limit))

        datasets_to_concat = [first]
        for path in paths[1:]:
            dataset = load_dataset(
                file_type,
                data_files={split: [path]},
                split=split,
                cache_dir=model_args.cache_dir,
            )
            if sample_limit is not None and len(dataset) > sample_limit:
                dataset = dataset.shuffle(seed=seed).select(range(sample_limit))

            datasets_to_concat.append(dataset)

        raw_datasets[split] = concatenate_datasets(datasets_to_concat).shuffle(
            seed=training_args.seed
        )

    dataset_dict = DatasetDict(raw_datasets)

    for split in dataset_dict.keys():
        dataset_dict[split] = dataset_dict[split].shuffle(seed=seed)

    # Rename text column to sentence1 if needed
    for split, dataset in dataset_dict.items():
        column_names = dataset.column_names
        if "sentence1" not in column_names:
            text_column = next(
                (candidate for candidate in ["text", "sentence", "document", "utterance"] if candidate in column_names),
                None,
            )
            if text_column is None:
                raise ValueError(
                    "Dataset must include a text column named 'sentence1' or one of "
                    "['text', 'sentence', 'document', 'utterance']."
                )
            dataset = dataset.rename_column(text_column, "sentence1")
        dataset_dict[split] = dataset

    return dataset_dict
