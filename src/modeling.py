from __future__ import annotations
from typing import Optional, List
import torch
from transformers import AutoModelForSequenceClassification, Trainer

# Optional LoRA/PEFT
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


class FocalBCEWithLogitsLoss(torch.nn.Module):
    """
    Focal loss on top of BCEWithLogits for multi-label classification.
    gamma > 0 focuses on hard examples; alpha balances positive/negative.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal = (self.alpha * (1 - pt).pow(self.gamma)) * bce
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


class WeightedTrainer(Trainer):
    """
    Trainer that supports per-class pos_weight for BCE or focal loss.
    Set either pos_weight (BCE) or focal_loss (FocalBCEWithLogitsLoss).
    """
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, focal_loss: Optional[FocalBCEWithLogitsLoss] = None, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight
        self.focal_loss = focal_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.focal_loss is not None:
            loss = self.focal_loss(logits, labels)
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None
            )
            loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def build_model(model_name: str, num_labels: int, gradient_checkpointing: bool, lora: bool,
                lora_r: int, lora_alpha: int, lora_dropout: float, lora_target_modules: List[str]):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft is not installed. Install with: pip install peft")
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model
