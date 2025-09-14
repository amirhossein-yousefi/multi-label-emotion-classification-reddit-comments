# sagemaker/entrypoint/inference.py
import os, json
import numpy as np
import torch
from typing import List, Dict, Any
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x))

def model_fn(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    labels_path = os.path.join(model_dir, "label_names.json")
    with open(labels_path, "r") as f:
        label_names = json.load(f)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return {"model": model, "tokenizer": tokenizer, "label_names": label_names, "device": device}

def _normalize_inputs(obj):
    """
    Accepts:
      - {"inputs": "text"} or {"inputs": ["t1", "t2"], "threshold": 0.5}
      - "text" or ["t1","t2"]
    Returns (list[str], threshold or None)
    """
    thr = None
    if isinstance(obj, dict):
        thr = obj.get("threshold")
        inputs = obj.get("inputs", obj.get("text", obj.get("data", "")))
    else:
        inputs = obj
    if isinstance(inputs, str):
        inputs = [inputs]
    if not isinstance(inputs, list):
        raise ValueError("Invalid input format")
    return inputs, thr

def input_fn(request_body, content_type="application/json"):
    if content_type == "application/json":
        obj = json.loads(request_body)
        return _normalize_inputs(obj)
    elif content_type == "text/plain":
        return _normalize_inputs(request_body)
    else:
        raise ValueError(f"Unsupported content_type: {content_type}")

def predict_fn(data, ctx):
    inputs, thr_override = data
    model = ctx["model"]; tokenizer = ctx["tokenizer"]; label_names = ctx["label_names"]; device = ctx["device"]

    enc = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits.detach().cpu().numpy()

    probs = sigmoid(logits)
    threshold = float(thr_override) if thr_override is not None else 0.5
    preds = (probs >= threshold).astype(int)
    results = []
    for i in range(len(inputs)):
        items = [{"label": label_names[j], "score": float(probs[i, j])} for j in range(len(label_names))]
        chosen = [label_names[j] for j in np.where(preds[i] == 1)[0]]
        results.append({"labels": chosen, "scores": items, "threshold": threshold})
    return results

def output_fn(prediction, accept="application/json"):
    if accept == "application/json":
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
