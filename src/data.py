from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding


def multi_hot(labels_list: List[List[int]], num_labels: int) -> List[List[float]]:
    out = []
    for labels in labels_list:
        vec = np.zeros(num_labels, dtype=np.float32)
        for idx in labels:
            if 0 <= idx < num_labels:
                vec[idx] = 1.0
        out.append(vec.tolist())
    return out


class MultiLabelCollator(DataCollatorWithPadding):
    """Ensures float32 labels for BCE* losses."""
    def __call__(self, features):
        labels = [f.pop("labels") for f in features]
        batch = super().__call__(features)
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch


def load_and_prepare_dataset(
    dataset_name: str,
    dataset_config: str,
    tokenizer: AutoTokenizer,
    text_column: str,
    labels_column: str,
    max_length: int,
) -> Tuple[DatasetDict, int, List[str]]:
    ds = load_dataset(dataset_name, dataset_config)  # splits: train/validation/test

    label_feature = ds["train"].features[labels_column].feature  # Sequence(ClassLabel)
    num_labels = label_feature.num_classes
    label_names = list(label_feature.names)

    def tokenize_fn(batch):
        enc = tokenizer(batch[text_column], truncation=True, max_length=max_length)
        enc["labels"] = multi_hot(batch[labels_column], num_labels)
        return enc

    keep = [text_column, labels_column]
    ds_enc = ds.map(tokenize_fn, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in keep])
    ds_enc = ds_enc.remove_columns([text_column])  # model doesn’t need raw text
    return ds_enc, num_labels, label_names
