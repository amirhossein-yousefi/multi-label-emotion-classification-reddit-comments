from __future__ import annotations
from typing import Dict, Union, Tuple
import numpy as np
from sklearn.metrics import (
    f1_score, average_precision_score, roc_auc_score
)
from transformers.trainer_utils import EvalPrediction


class MetricComputer:
    """Compute multi-label metrics using a fixed decision threshold."""
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, eval_pred: Union[Tuple[np.ndarray, np.ndarray], EvalPrediction]) -> Dict[str, float]:
        if isinstance(eval_pred, EvalPrediction):
            logits = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            logits, labels = eval_pred

        probs = 1.0 / (1.0 + np.exp(-logits))
        y_pred = (probs >= self.threshold).astype(int)

        metrics = {
            "f1_micro": f1_score(labels, y_pred, average="micro", zero_division=0),
            "f1_macro": f1_score(labels, y_pred, average="macro", zero_division=0),
            "f1_samples": f1_score(labels, y_pred, average="samples", zero_division=0),
        }

        # AP / ROC-AUC
        try:
            metrics["ap_micro"] = float(average_precision_score(labels, probs, average="micro"))
            metrics["ap_macro"] = float(average_precision_score(labels, probs, average="macro"))
        except Exception:
            metrics["ap_micro"] = float("nan")
            metrics["ap_macro"] = float("nan")
        try:
            metrics["auc_micro"] = float(roc_auc_score(labels, probs, average="micro"))
            metrics["auc_macro"] = float(roc_auc_score(labels, probs, average="macro"))
        except Exception:
            metrics["auc_micro"] = float("nan")
            metrics["auc_macro"] = float("nan")
        return metrics
