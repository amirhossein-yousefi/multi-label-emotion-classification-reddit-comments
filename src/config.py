from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json
import os

try:
    import yaml  # optional
except Exception:
    yaml = None


@dataclass
class TrainConfig:
    seed: int = 42
    output_dir: str = "outputs/goemotions_roberta"

    dataset_name: str = "go_emotions"
    dataset_config: str = "simplified"
    text_column: str = "text"
    labels_column: str = "labels"
    max_length: int = 192

    model_name: str = "roberta-base"
    batch_size: int = 16
    eval_batch_size: int = 32
    lr: float = 2e-5
    epochs: int = 5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    grad_accum: int = 1
    patience: int = 2
    gradient_checkpointing: bool = False

    use_bf16_if_available: bool = True
    use_fp16_if_available: bool = True
    eval_every_steps: int = 0  # 0 = epoch-based

    lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["query", "value"])

    loss_type: str = "bce"       # "bce" | "focal"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    threshold_grid: Dict[str, float] = field(default_factory=lambda: {"start": 0.05, "stop": 0.95, "step": 0.01})

    log_level: str = "INFO"
    save_total_limit: int = 2

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        return cls(**d)

    @classmethod
    def from_file(cls, path: str) -> "TrainConfig":
        with open(path, "r") as f:
            if path.endswith((".yml", ".yaml")):
                if yaml is None:
                    raise RuntimeError("PyYAML not installed. Install pyyaml or use a JSON config.")
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            if path.endswith((".yml", ".yaml")) and yaml is not None:
                yaml.safe_dump(self.to_dict(), f, sort_keys=False)
            else:
                json.dump(self.to_dict(), f, indent=2)
