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


def compute_ldam_margins(class_counts: Sequence[int], *, max_margin: float = 0.5) -> list[float]:
    """Return the per-class margins used by LDAM (larger for rarer classes)."""
    counts = list(class_counts)
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("每个类别的样本数必须为正数")
    if max_margin <= 0.0:
        raise ValueError("LDAM最大边距必须为正数")
    raw_margins = [count ** -0.25 for count in counts]
    scale = max_margin / max(raw_margins)
    return [margin * scale for margin in raw_margins]


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


class LDAMDRWClassificationLoss:
    """LDAM margin loss with deferred class reweighting during late epochs."""

    def __init__(
        self,
        margins: Sequence[float],
        deferred_weights: Sequence[float],
        *,
        scale: float = 30.0,
        drw_start_epoch: int = 10,
        label_smoothing: float = 0.0,
    ) -> None:
        if scale <= 0.0:
            raise ValueError("LDAM缩放系数必须为正数")
        if drw_start_epoch < 0:
            raise ValueError("DRW起始epoch不能小于0")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing必须在[0, 1)之间")
        self.margins = torch.tensor(list(margins), dtype=torch.float32)
        self.deferred_weights = torch.tensor(list(deferred_weights), dtype=torch.float32)
        self.scale = scale
        self.drw_start_epoch = drw_start_epoch
        self.label_smoothing = label_smoothing
        self.current_epoch = 0

    def __call__(
        self, preds: Any, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = preds[1] if isinstance(preds, (list, tuple)) else preds
        targets = batch["cls"]
        target_indexes = targets.view(-1, 1)
        margins = self.margins.to(device=logits.device, dtype=logits.dtype)[targets].view(-1, 1)
        adjusted_logits = logits.clone()
        adjusted_logits.scatter_add_(1, target_indexes, -margins)
        weights = None
        if self.current_epoch >= self.drw_start_epoch:
            weights = self.deferred_weights.to(device=logits.device, dtype=logits.dtype)
        loss = F.cross_entropy(
            adjusted_logits * self.scale,
            targets,
            weight=weights,
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
        ldam_max_margin=0.5,
        ldam_scale=30.0,
        drw_start_epoch=10,
        label_smoothing=label_smoothing,
        image_transform=image_transform,
    )


def make_long_tail_classification_trainer(
    *,
    strategy: str,
    power: float,
    cap: float,
    logit_adjustment_tau: float,
    ldam_max_margin: float,
    ldam_scale: float,
    drw_start_epoch: int,
    label_smoothing: float,
    image_transform: str = "default",
) -> type:
    if strategy not in {"weighted_ce", "logit_adjusted", "ldam_drw"}:
        raise ValueError("长尾策略仅支持weighted_ce、logit_adjusted或ldam_drw")
    if image_transform not in {"default", "letterbox"}:
        raise ValueError("image_transform仅支持default或letterbox")
    if logit_adjustment_tau < 0.0:
        raise ValueError("logit adjustment的tau不能小于0")
    if ldam_max_margin <= 0.0 or ldam_scale <= 0.0 or drw_start_epoch < 0:
        raise ValueError("LDAM-DRW参数不合法")
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

        def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
            batch = super().preprocess_batch(batch)
            criterion = unwrap_model(self.model).criterion
            if isinstance(criterion, LDAMDRWClassificationLoss):
                criterion.current_epoch = int(self.epoch) + 1
            return batch

        def set_class_weights(self) -> None:
            counts_by_index = Counter(int(sample[1]) for sample in self.train_loader.dataset.samples)
            class_count = int(self.data["nc"])
            counts = [counts_by_index[index] for index in range(class_count)]
            weights = compute_class_weights(counts, power=power, cap=cap)
            adjustments = compute_logit_adjustments(counts, tau=logit_adjustment_tau)
            margins = compute_ldam_margins(counts, max_margin=ldam_max_margin)
            if strategy == "weighted_ce":
                criterion = WeightedClassificationLoss(weights, label_smoothing=label_smoothing)
            elif strategy == "logit_adjusted":
                criterion = LogitAdjustedClassificationLoss(
                    adjustments, label_smoothing=label_smoothing
                )
            else:
                criterion = LDAMDRWClassificationLoss(
                    margins,
                    weights,
                    scale=ldam_scale,
                    drw_start_epoch=drw_start_epoch,
                    label_smoothing=label_smoothing,
                )
            unwrap_model(self.model).criterion = criterion
            names = self.data["names"]
            report = {
                "strategy": strategy,
                "power": power,
                "cap": cap,
                "logit_adjustment_tau": logit_adjustment_tau,
                "ldam_max_margin": ldam_max_margin,
                "ldam_scale": ldam_scale,
                "drw_start_epoch": drw_start_epoch,
                "label_smoothing": label_smoothing,
                "image_transform": image_transform,
                "classes": [
                    {
                        "index": index,
                        "label": str(names[index]),
                        "count": counts[index],
                        "weight": weights[index],
                        "logit_adjustment": adjustments[index],
                        "ldam_margin": margins[index],
                    }
                    for index in range(class_count)
                ],
            }
            Path(self.save_dir, "class_weights.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    return WeightedClassificationTrainer
