from __future__ import annotations
import argparse
import csv
import os
from typing import Dict, Any

import numpy as np
import torch
from datasets import DatasetDict
from sklearn.metrics import (
    f1_score, average_precision_score, roc_auc_score,
    precision_recall_fscore_support, classification_report
)
from transformers import (
    AutoTokenizer, TrainingArguments, EarlyStoppingCallback
)

from src.config import TrainConfig
from src.utils import setup_logging, set_seed, compute_pos_weight, save_json, get_versions_metadata, sigmoid_np
from src.data import load_and_prepare_dataset, MultiLabelCollator
from src.metrics import MetricComputer
from src.threshold import best_threshold_by_micro_f1
from src.modeling import build_model, WeightedTrainer, FocalBCEWithLogitsLoss

# Ensure CuBLAS determinism workspace (second line of defense, but best is to set this in train.py before imports)
if torch.cuda.is_available() and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# Prefer warnings over hard failure if some ops can't be deterministic.
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
# Older PyTorch without warn_only still works, but may raise at runtime if an op is non-deterministic.
    torch.use_deterministic_algorithms(True)
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-label text classification training")
    p.add_argument("--config", type=str, default="../configs/base.yaml", help="Path to YAML/JSON config")
    # Optional quick overrides:
    p.add_argument("--output_dir", type=str)
    p.add_argument("--model_name", type=str)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--eval_batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--lora", action="store_true")
    return p.parse_args()


def merge_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    # Minimal overrides for convenience
    if args.output_dir: cfg.output_dir = args.output_dir
    if args.model_name: cfg.model_name = args.model_name
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.eval_batch_size is not None: cfg.eval_batch_size = args.eval_batch_size
    if args.lr is not None: cfg.lr = args.lr
    if args.lora: cfg.lora = True
    return cfg


def main():
    args = parse_args()
    cfg = TrainConfig.from_file(args.config)
    cfg = merge_overrides(cfg, args)

    os.makedirs(cfg.output_dir, exist_ok=True)
    logger = setup_logging(cfg.output_dir, cfg.log_level)
    set_seed(cfg.seed)
    save_json(os.path.join(cfg.output_dir, "config_resolved.json"), cfg.to_dict())
    save_json(os.path.join(cfg.output_dir, "run_metadata.json"), get_versions_metadata())

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    logger.info("Loading and tokenizing dataset...")
    ds_enc, num_labels, label_names = load_and_prepare_dataset(
        cfg.dataset_name, cfg.dataset_config, tokenizer, cfg.text_column, cfg.labels_column, cfg.max_length
    )

    # pos_weight from train labels
    y_train = np.array(ds_enc["train"]["labels"], dtype=np.float32)
    pos_weight = compute_pos_weight(y_train)

    # Build model
    logger.info("Loading model...")
    model = build_model(cfg.model_name, num_labels, cfg.gradient_checkpointing, cfg.lora,
                        cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout, cfg.lora_target_modules)

    # Collator & metrics
    collator = MultiLabelCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    metric_cb = MetricComputer(threshold=0.5)

    # AMP options
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and cfg.use_bf16_if_available
    fp16 = torch.cuda.is_available() and (not bf16) and cfg.use_fp16_if_available

    eval_strategy = "epoch" if cfg.eval_every_steps <= 0 else "steps"

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy=eval_strategy,
        save_strategy=eval_strategy,
        eval_steps=cfg.eval_every_steps if cfg.eval_every_steps > 0 else None,
        save_steps=cfg.eval_every_steps if cfg.eval_every_steps > 0 else None,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=cfg.save_total_limit,
        fp16=fp16,
        bf16=bf16,
        report_to=["tensorboard"],
        logging_dir=os.path.join(cfg.output_dir, "logs"),
    )

    # Loss selection
    focal_loss = None
    if cfg.loss_type.lower() == "focal":
        focal_loss = FocalBCEWithLogitsLoss(gamma=cfg.focal_gamma, alpha=cfg.focal_alpha)
        pos_weight_tensor = None
        logger.info(f"Using FocalBCEWithLogitsLoss(gamma={cfg.focal_gamma}, alpha={cfg.focal_alpha})")
    else:
        pos_weight_tensor = pos_weight
        logger.info("Using BCEWithLogitsLoss with pos_weight")

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_enc["train"],
        eval_dataset=ds_enc["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=metric_cb,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience)],
        pos_weight=pos_weight_tensor,
        focal_loss=focal_loss,
    )

    logger.info("Starting training...")
    trainer.train()

    # Threshold tuning on validation
    logger.info("Tuning decision threshold (validation)...")
    val_out = trainer.predict(ds_enc["validation"])
    val_probs = sigmoid_np(val_out.predictions)
    best_t = best_threshold_by_micro_f1(
        val_out.label_ids,
        val_probs,
        start=cfg.threshold_grid["start"],
        stop=cfg.threshold_grid["stop"],
        step=cfg.threshold_grid["step"],
    )
    logger.info(f"Best threshold (micro-F1 on val): {best_t:.2f}")
    save_json(os.path.join(cfg.output_dir, "threshold.json"), {"threshold": best_t})

    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_out = trainer.predict(ds_enc["test"])
    test_probs = sigmoid_np(test_out.predictions)
    y_pred = (test_probs >= best_t).astype(int)
    y_true = test_out.label_ids

    results: Dict[str, Any] = {
        "threshold": float(best_t),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_samples": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "ap_micro": float(average_precision_score(y_true, test_probs, average="micro")),
        "ap_macro": float(average_precision_score(y_true, test_probs, average="macro")),
    }
    try:
        results["auc_micro"] = float(roc_auc_score(y_true, test_probs, average="micro"))
        results["auc_macro"] = float(roc_auc_score(y_true, test_probs, average="macro"))
    except Exception:
        results["auc_micro"] = float("nan")
        results["auc_macro"] = float("nan")

    # Log & persist metrics
    logger.info("=== Test Metrics ===")
    for k, v in results.items():
        logger.info(f"{k:>12}: {v}")

    save_json(os.path.join(cfg.output_dir, "test_metrics.json"), results)

    # Per-label metrics
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    per_label_rows = []
    for i, name in enumerate(label_names):
        per_label_rows.append({
            "label": name,
            "support": int(support[i]),
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
        })
    # Save as CSV for easy inspection
    csv_path = os.path.join(cfg.output_dir, "per_label_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "support", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(per_label_rows)

    # Full text report
    try:
        report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
        with open(os.path.join(cfg.output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)
    except Exception:
        pass

    # Save model/tokenizer + labels
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # Save label names for inference
    save_json(os.path.join(cfg.output_dir, "label_names.json"), label_names)

    logger.info(f"Run complete. Artifacts saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
