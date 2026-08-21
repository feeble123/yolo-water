from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


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


def make_weighted_classification_trainer(
    *, power: float, cap: float, label_smoothing: float
) -> type:
    from ultralytics.models.yolo.classify import ClassificationTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    class WeightedClassificationTrainer(ClassificationTrainer):
        def set_class_weights(self) -> None:
            counts_by_index = Counter(int(sample[1]) for sample in self.train_loader.dataset.samples)
            class_count = int(self.data["nc"])
            counts = [counts_by_index[index] for index in range(class_count)]
            weights = compute_class_weights(counts, power=power, cap=cap)
            unwrap_model(self.model).criterion = WeightedClassificationLoss(
                weights, label_smoothing=label_smoothing
            )
            names = self.data["names"]
            report = {
                "strategy": "weighted_cross_entropy",
                "power": power,
                "cap": cap,
                "label_smoothing": label_smoothing,
                "classes": [
                    {
                        "index": index,
                        "label": str(names[index]),
                        "count": counts[index],
                        "weight": weights[index],
                    }
                    for index in range(class_count)
                ],
            }
            Path(self.save_dir, "class_weights.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    return WeightedClassificationTrainer
