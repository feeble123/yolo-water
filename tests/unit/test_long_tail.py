from __future__ import annotations

import pytest
import torch

from water_agent.vision.long_tail import (
    LogitAdjustedClassificationLoss,
    WeightedClassificationLoss,
    compute_class_weights,
    compute_logit_adjustments,
)


def test_compute_class_weights_uses_sqrt_inverse_frequency_and_cap() -> None:
    weights = compute_class_weights([100, 25, 1], power=0.5, cap=5.0)

    assert weights == pytest.approx([1.0, 2.0, 5.0])


def test_compute_class_weights_rejects_empty_classes() -> None:
    with pytest.raises(ValueError, match="样本数必须为正数"):
        compute_class_weights([10, 0], power=0.5)


def test_weighted_loss_matches_pytorch_cross_entropy() -> None:
    logits = torch.tensor([[2.0, 0.1], [0.2, 1.3]])
    batch = {"cls": torch.tensor([0, 1])}
    criterion = WeightedClassificationLoss([1.0, 3.0])

    loss, items = criterion(logits, batch)
    expected = torch.nn.functional.cross_entropy(
        logits, batch["cls"], weight=torch.tensor([1.0, 3.0])
    )

    assert loss.item() == pytest.approx(expected.item())
    assert items["loss"].item() == pytest.approx(expected.item())


def test_logit_adjustments_follow_class_priors() -> None:
    adjustments = compute_logit_adjustments([90, 10], tau=0.5)

    assert adjustments[0] == pytest.approx(0.5 * torch.log(torch.tensor(0.9)).item())
    assert adjustments[1] == pytest.approx(0.5 * torch.log(torch.tensor(0.1)).item())
    assert adjustments[0] > adjustments[1]


def test_logit_adjusted_loss_matches_adjusted_cross_entropy() -> None:
    logits = torch.tensor([[2.0, 0.1], [0.2, 1.3]])
    batch = {"cls": torch.tensor([0, 1])}
    criterion = LogitAdjustedClassificationLoss([0.0, -1.0])

    loss, items = criterion(logits, batch)
    expected = torch.nn.functional.cross_entropy(logits + torch.tensor([0.0, -1.0]), batch["cls"])

    assert loss.item() == pytest.approx(expected.item())
    assert items["loss"].item() == pytest.approx(expected.item())
