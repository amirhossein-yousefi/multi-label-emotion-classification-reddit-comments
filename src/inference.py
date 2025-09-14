from __future__ import annotations
import argparse
import json
import os
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_threshold(path: str) -> float:
    with open(path, "r", encoding="utf-8") as f:
        return float(json.load(f)["threshold"])


def load_label_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return list(json.load(f))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def predict(texts: List[str], model_dir: str, threshold: float | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if threshold is None and os.path.exists(os.path.join(model_dir, "threshold.json")):
        threshold = load_threshold(os.path.join(model_dir, "threshold.json"))
    labels = load_label_names(os.path.join(model_dir, "label_names.json"))

    enc = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits.detach().cpu().numpy()
    probs = sigmoid(logits)

    preds = (probs >= (threshold if threshold is not None else 0.5)).astype(int)

    outputs = []
    for i, text in enumerate(texts):
        chosen = [labels[j] for j, v in enumerate(preds[i]) if v == 1]
        outputs.append({"text": text, "labels": chosen, "probs": probs[i].tolist()})
    return outputs


def main():
    ap = argparse.ArgumentParser(description="Run inference with a trained emoclass model")
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--text", type=str, nargs="+", help="One or more texts to classify")
    ap.add_argument("--file", type=str, help="Optional path to a text file (one example per line)")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    texts: List[str] = []
    if args.text:
        texts.extend(args.text)
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts.extend([line.strip() for line in f if line.strip()])

    if not texts:
        raise SystemExit("Provide --text or --file")

    outputs = predict(texts, args.model_dir, threshold=args.threshold)
    for out in outputs:
        print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
