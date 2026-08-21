from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from water_agent.vision.full_frame import build_full_frame_transform


def compute_class_weights(
    class_counts: Sequence[int], *, power: float = 0.5, cap: float = 8.0
) -> list[float]:
    counts = list(class_counts)
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("每个类别的样本数必须为正数")
    if not 0.0 <= power <= 1.0:
        raise ValueError("power必须在0到1之间")
    if cap < 1.0:
        raise ValueError("cap不能小于1")
    majority_count = max(counts)
    return [min((majority_count / count) ** power, cap) for count in counts]


def compute_logit_adjustments(class_counts: Sequence[int], *, tau: float) -> list[float]:
    """Return tau * log(class prior) for long-tail logit-adjusted training."""
    counts = list(class_counts)
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("每个类别的样本数必须为正数")
    if tau < 0.0:
        raise ValueError("logit adjustment的tau不能小于0")
    total = float(sum(counts))
    return [tau * float(torch.log(torch.tensor(count / total)).item()) for count in counts]


class WeightedClassificationLoss:
    def __init__(self, weights: Sequence[float], label_smoothing: float = 0.0) -> None:
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing必须在[0, 1)之间")
        self.weights = torch.tensor(list(weights), dtype=torch.float32)
        self.label_smoothing = label_smoothing

    def __call__(
        self, preds: Any, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(
            logits,
            batch["cls"],
            weight=self.weights.to(device=logits.device, dtype=logits.dtype),
            reduction="mean",
            label_smoothing=self.label_smoothing,
        )
        return loss, {"loss": loss.detach()}


class LogitAdjustedClassificationLoss:
    """Cross entropy with class-prior logit adjustment during training only."""

    def __init__(self, adjustments: Sequence[float], label_smoothing: float = 0.0) -> None:
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing必须在[0, 1)之间")
        self.adjustments = torch.tensor(list(adjustments), dtype=torch.float32)
        self.label_smoothing = label_smoothing

    def __call__(
        self, preds: Any, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = preds[1] if isinstance(preds, (list, tuple)) else preds
        adjusted_logits = logits + self.adjustments.to(device=logits.device, dtype=logits.dtype)
        loss = F.cross_entropy(
            adjusted_logits,
            batch["cls"],
            reduction="mean",
            label_smoothing=self.label_smoothing,
        )
        return loss, {"loss": loss.detach()}


def make_weighted_classification_trainer(
    *, power: float, cap: float, label_smoothing: float, image_transform: str = "default"
) -> type:
    return make_long_tail_classification_trainer(
        strategy="weighted_ce",
        power=power,
        cap=cap,
        logit_adjustment_tau=0.0,
        label_smoothing=label_smoothing,
        image_transform=image_transform,
    )


def make_long_tail_classification_trainer(
    *,
    strategy: str,
    power: float,
    cap: float,
    logit_adjustment_tau: float,
    label_smoothing: float,
    image_transform: str = "default",
) -> type:
    if strategy not in {"weighted_ce", "logit_adjusted"}:
        raise ValueError("长尾策略仅支持weighted_ce或logit_adjusted")
    if image_transform not in {"default", "letterbox"}:
        raise ValueError("image_transform仅支持default或letterbox")
    if logit_adjustment_tau < 0.0:
        raise ValueError("logit adjustment的tau不能小于0")
    from ultralytics.data import ClassificationDataset
    from ultralytics.models.yolo.classify import ClassificationTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    class WeightedClassificationTrainer(ClassificationTrainer):
        def build_dataset(self, img_path: str, mode: str = "train", batch: Any = None):
            if image_transform == "default":
                return super().build_dataset(img_path, mode, batch)
            dataset = ClassificationDataset(
                root=img_path, args=self.args, augment=mode == "train", prefix=mode
            )
            dataset.torch_transforms = build_full_frame_transform(
                size=int(self.args.imgsz),
                train=mode == "train",
                horizontal_flip=float(self.args.fliplr),
            )
            return dataset

        def set_class_weights(self) -> None:
            counts_by_index = Counter(int(sample[1]) for sample in self.train_loader.dataset.samples)
            class_count = int(self.data["nc"])
            counts = [counts_by_index[index] for index in range(class_count)]
            weights = compute_class_weights(counts, power=power, cap=cap)
            adjustments = compute_logit_adjustments(counts, tau=logit_adjustment_tau)
            if strategy == "weighted_ce":
                criterion = WeightedClassificationLoss(weights, label_smoothing=label_smoothing)
            else:
                criterion = LogitAdjustedClassificationLoss(
                    adjustments, label_smoothing=label_smoothing
                )
            unwrap_model(self.model).criterion = criterion
            names = self.data["names"]
            report = {
                "strategy": strategy,
                "power": power,
                "cap": cap,
                "logit_adjustment_tau": logit_adjustment_tau,
                "label_smoothing": label_smoothing,
                "image_transform": image_transform,
                "classes": [
                    {
                        "index": index,
                        "label": str(names[index]),
                        "count": counts[index],
                        "weight": weights[index],
                        "logit_adjustment": adjustments[index],
                    }
                    for index in range(class_count)
                ],
            }
            Path(self.save_dir, "class_weights.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    return WeightedClassificationTrainer
