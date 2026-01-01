"""Label mapping and preparation utilities."""
import json
import os
import logging
from typing import Dict, List, Tuple, Optional
from datasets import DatasetDict
from utils.src.data.preprocessing import parse_dict

logger = logging.getLogger(__name__)


def prepare_label_mappings(
    raw_datasets: DatasetDict,
    model_args,
    data_args,
) -> Tuple[
    DatasetDict,
    List[str],
    Dict[int, str],
    Dict[str, int],
    List[str],
    Optional[Dict],
    Dict[str, Dict],
    Dict[str, Dict[int, str]],
]:
    """
    Prepare label mappings for the datasets.
    
    Args:
        raw_datasets: Raw datasets
        model_args: Model arguments
        data_args: Data arguments
        
    Returns:
        Tuple containing:
        - updated_datasets
        - labels
        - id2label
        - label2id
        - aspect_key
        - classifier_configs
        - classifier_configs_for_trainer
        - label_name_mappings
    """
    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")

    labels = [
        key
        for key in raw_datasets["validation"].features.keys()
        if key not in {"sentence1", "sentence2", "sentence3"}
    ]

    id2label = {i: aspect for i, aspect in enumerate(list(labels))}
    label2id = {aspect: i for i, aspect in enumerate(list(labels))}

    if model_args.classifier_configs is not None:
        if os.path.exists(model_args.classifier_configs):
            raw_configs = json.load(open(model_args.classifier_configs))
        else:
            raw_configs = parse_dict(model_args.classifier_configs)
        
        classifier_configs = {}
        aspect_key = []
        for key, value in raw_configs.items():
            aspect_key.append(key)
            if isinstance(value, list):
                for cfg in value:
                    head_name = cfg.get("head_name")
                    if not head_name:
                        raise ValueError(f"head_name is required for multiple classifiers on key '{key}'")
                    cfg["dataset_column"] = key
                    classifier_configs[head_name] = cfg
            else:
                value["dataset_column"] = key
                head_name = value.get("head_name", key)
                classifier_configs[head_name] = value
    else:
        classifier_configs = None
        aspect_key = getattr(model_args, 'aspect_key', [])

    if aspect_key is None:
        aspect_key = []
    elif isinstance(aspect_key, str):
        aspect_key = [aspect_key]
    else:
        aspect_key = list(aspect_key)

    # Rename label columns
    label_candidates = ["labels", "label", "target"]
    updated_datasets = raw_datasets
    for split, dataset in updated_datasets.items():
        for head in aspect_key:
            if head in dataset.column_names:
                continue
            renamed = False
            for candidate in label_candidates:
                if candidate in dataset.column_names:
                    dataset = dataset.rename_column(candidate, head)
                    renamed = True
                    break
            if not renamed:
                raise ValueError(
                    f"Expected label column for head '{head}' not found in dataset columns: {dataset.column_names}"
                )
        updated_datasets[split] = dataset

    # Build label name mappings
    label_name_mappings: Dict[str, Dict[int, str]] = {}

    for head in aspect_key:
        sample_values: List = []
        for dataset in updated_datasets.values():
            if head in dataset.column_names:
                values = [v for v in dataset[head] if v is not None]
                if values:
                    sample_values.extend(values)
                    break
        if not sample_values:
            continue
        if isinstance(sample_values[0], str):
            unique_values_set = {
                v
                for ds in updated_datasets.values()
                if head in ds.column_names
                for v in ds[head]
            }
            
            # Sort by angle if angle_map is available in classifier_configs
            angle_map = None
            if classifier_configs:
                for cfg_name, cfg in classifier_configs.items():
                    if cfg.get("dataset_column") == head and "angle_map" in cfg:
                        angle_map = cfg["angle_map"]
                        break
            
            if angle_map:
                # Create label-to-angle mapping
                label_to_angle = {label_name: float(angle) for angle, label_name in angle_map.items()}
                # Sort by angle value
                unique_values = sorted(
                    unique_values_set,
                    key=lambda x: (x not in label_to_angle, label_to_angle.get(x, 0))
                )
            else:
                unique_values = list(unique_values_set)
            
            label_to_id_map = {label: idx for idx, label in enumerate(unique_values)}
            label_name_mappings[head] = {
                idx: label for label, idx in label_to_id_map.items()
            }

            def _encode_labels(example):
                value = example.get(head)
                if value is not None and value in label_to_id_map:
                    example[head] = label_to_id_map[value]
                return example

            updated_datasets = updated_datasets.map(
                _encode_labels,
                desc=f"Encoding labels for {head}",
                load_from_cache_file=not data_args.overwrite_cache,
            )

    classifier_configs_for_trainer = (
        classifier_configs
        if classifier_configs is not None
        else {head: {"objective": getattr(model_args, 'objective', 'SINCERE')} for head in aspect_key}
    )

    return (
        updated_datasets,
        labels,
        id2label,
        label2id,
        aspect_key,
        classifier_configs,
        classifier_configs_for_trainer,
        label_name_mappings,
    )
