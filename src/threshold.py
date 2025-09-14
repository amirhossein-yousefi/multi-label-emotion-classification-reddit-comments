from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score


def best_threshold_by_micro_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.01,
) -> float:
    thresholds = np.arange(start, stop + 1e-9, step)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t
