# sagemaker/entrypoint/train.py
import argparse, os, json, random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from datasets import load_dataset

# ---------------- utilities ----------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def binarize(probs: np.ndarray, thr: float) -> np.ndarray:
    return (probs >= thr).astype(int)

def infer_label_matrix(df: pd.DataFrame, text_col: str, labels_col: Optional[str], label_cols: Optional[List[str]]):
    """
    Returns (y, label_names). Accepts:
      - list/str in a 'labels' column (JSON list or `;`-separated),
      - or explicitly-provided one-hot columns via label_cols,
      - or falls back to one-hot columns = all columns except text_col.
    """
    if label_cols:
        label_names = label_cols
        y = df[label_names].astype(int).values
        return y, label_names

    if labels_col and labels_col in df.columns:
        parsed = []
        for v in df[labels_col].fillna("").astype(str):
            v = v.strip()
            if v.startswith("["):  # JSON list of labels
                try:
                    lst = json.loads(v)
                except Exception:
                    lst = []
            else:
                lst = [s.strip() for s in v.split(";") if s.strip()]
            parsed.append(lst)
        # collect unique label names in order of appearance
        label_names: List[str] = []
        for lst in parsed:
            for l in lst:
                if l not in label_names:
                    label_names.append(l)
        idx = {l:i for i,l in enumerate(label_names)}
        y = np.zeros((len(parsed), len(label_names)), dtype=int)
        for r, lst in enumerate(parsed):
            for l in lst:
                if l in idx:
                    y[r, idx[l]] = 1
        return y, label_names

    # fallback: assume all columns except text are one-hots
    candidates = [c for c in df.columns if c != text_col]
    if not candidates:
        raise ValueError("Could not infer label columns. Provide label_cols or a 'labels' column.")
    label_names = candidates
    y = df[label_names].astype(int).values
    return y, label_names

@dataclass
class TextMultiLabelDataset:
    encodings: Dict[str, Any]
    labels: np.ndarray

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx]).float()
        return item

def build_from_dataframe(df: pd.DataFrame, tokenizer, text_col: str, max_len: int, labels: np.ndarray) -> TextMultiLabelDataset:
    enc = tokenizer(df[text_col].tolist(), truncation=True, padding=False, max_length=max_len)
    return TextMultiLabelDataset(enc, labels)

def make_metrics(num_labels: int, threshold: float):
    def compute(eval_pred):
        # Compatible with EvalPrediction or tuple
        logits = getattr(eval_pred, "predictions", None)
        labels = getattr(eval_pred, "label_ids", None)
        if logits is None:
            logits, labels = eval_pred
        probs = sigmoid(logits)
        preds = binarize(probs, threshold)
        return {
            "f1_micro": float(f1_score(labels, preds, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "precision_micro": float(precision_score(labels, preds, average="micro", zero_division=0)),
            "recall_micro": float(recall_score(labels, preds, average="micro", zero_division=0)),
        }
    return compute

# ---------------- main ----------------

def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--labels_column", type=str, default="labels")
    parser.add_argument("--label_cols", type=str, default="")  # comma-separated list
    parser.add_argument("--use_hf_goemotions", action="store_true", help="If true, download go_emotions (simplified) from Hugging Face.")
    parser.add_argument("--include_neutral", action="store_true", help="Include 'neutral' label when using go_emotions.")
    parser.add_argument("--val_ratio", type=float, default=0.05)

    # model/hparams
    parser.add_argument("--model_name_or_path", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)

    # SageMaker env defaults
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    parser.add_argument("--train_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", ""))
    parser.add_argument("--validation_dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", ""))

    args = parser.parse_args()
    set_seed(args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    # ----------- Load data -----------
    if args.use_hf_goemotions:
        # GoEmotions: Reddit comments multi-label emotions (27 emotions + optional neutral)
        ds = load_dataset("go_emotions", "simplified")
        all_label_names: List[str] = ds["train"].features["labels"].feature.names  # includes 'neutral'
        label_names = list(all_label_names)
        if not args.include_neutral and "neutral" in label_names:
            label_names.remove("neutral")

        def to_df(split: str) -> pd.DataFrame:
            texts = ds[split]["text"]
            labels_list = ds[split]["labels"]
            rows = []
            for t, lst in zip(texts, labels_list):
                row = {"text": t}
                row_labels = [0]*len(label_names)
                for li in lst:
                    name = all_label_names[li]
                    if name in label_names:
                        row_labels[label_names.index(name)] = 1
                row.update({name: v for name, v in zip(label_names, row_labels)})
                rows.append(row)
            return pd.DataFrame(rows)

        train_df = to_df("train")
        val_df = to_df("validation")

        y_train = train_df[label_names].values
        y_val   = val_df[label_names].values
        train_ds = build_from_dataframe(train_df[["text"] + label_names], tokenizer, "text", args.max_length, y_train)
        val_ds   = build_from_dataframe(val_df[["text"] + label_names], tokenizer, "text", args.max_length, y_val)
    else:
        # Load CSVs from SageMaker channels if provided; else error.
        def find_csv(d: str):
            if not d or not os.path.isdir(d): return None
            for name in ["train.csv", "training.csv", "data.csv"]:
                p = os.path.join(d, name)
                if os.path.exists(p): return p
            # otherwise first csv in dir
            for f in os.listdir(d):
                if f.lower().endswith(".csv"): return os.path.join(d, f)
            return None

        train_csv = find_csv(args.train_dir)
        if not train_csv:
            raise RuntimeError("No training data found. Either pass --use_hf_goemotions or provide a CSV in the train channel.")
        train_df = pd.read_csv(train_csv)

        val_csv = find_csv(args.validation_dir)
        if val_csv:
            val_df = pd.read_csv(val_csv)
        else:
            # small split for validation
            msk = np.random.rand(len(train_df)) > args.val_ratio
            val_df = train_df[~msk].reset_index(drop=True)
            train_df = train_df[msk].reset_index(drop=True)

        label_cols = [s.strip() for s in args.label_cols.split(",") if s.strip()] if args.label_cols else None
        y_train, label_names = infer_label_matrix(train_df, args.text_column, args.labels_column, label_cols)
        y_val, _ = infer_label_matrix(val_df, args.text_column, args.labels_column, label_cols)

        train_ds = build_from_dataframe(train_df[[args.text_column]], tokenizer, args.text_column, args.max_length, y_train)
        val_ds   = build_from_dataframe(val_df[[args.text_column]], tokenizer, args.text_column, args.max_length, y_val)

    num_labels = len(label_names)

    # ----------- Model -----------
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    config.problem_type = "multi_label_classification"  # enables BCEWithLogitsLoss in HF heads

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    metrics_fn = make_metrics(num_labels=num_labels, threshold=args.threshold)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.grad_accum_steps,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        logging_steps=50,
        seed=args.seed,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=metrics_fn
    )

    trainer.train()

    # Save final artifacts to SageMaker model dir so SageMaker uploads them to S3
    os.makedirs(args.model_dir, exist_ok=True)
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    with open(os.path.join(args.model_dir, "label_names.json"), "w") as f:
        json.dump(label_names, f)

if __name__ == "__main__":
    main()
