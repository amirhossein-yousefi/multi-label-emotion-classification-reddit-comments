from __future__ import annotations
import json
import logging
import os
import random
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import transformers
import datasets
import sklearn


def setup_logging(output_dir: str, level: str = "INFO") -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training.log")

    logger = logging.getLogger("emoclass")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("Logging initialized.")
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def get_versions_metadata() -> Dict[str, Any]:
    def git_rev() -> Optional[str]:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        except Exception:
            return None

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "sklearn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "git_commit": git_rev(),
    }


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def compute_pos_weight(train_label_vectors: np.ndarray) -> torch.Tensor:
    pos = train_label_vectors.sum(axis=0)
    n = train_label_vectors.shape[0]
    neg = n - pos
    w = np.where(pos > 0, neg / np.maximum(pos, 1e-9), 1.0)
    return torch.tensor(w, dtype=torch.float32)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
