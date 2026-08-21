from __future__ import annotations

import numpy as np
import pytest

from water_agent.vision.evaluate import calculate_majority_baseline, calculate_metrics


def test_calculate_metrics_for_perfect_predictions() -> None:
    labels = ["甲", "乙", "丙"]
    truth = ["甲", "乙", "丙", "甲"]
    probabilities = np.asarray(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
        ]
    )

    metrics = calculate_metrics(truth, probabilities, labels)

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["top2_accuracy"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_majority_baseline_exposes_imbalanced_accuracy() -> None:
    labels = ["少数甲", "多数", "少数乙"]
    truth = ["多数", "多数", "多数", "少数甲", "少数乙"]

    baseline = calculate_majority_baseline(truth, labels)

    assert baseline["majority_label"] == "多数"
    assert baseline["accuracy"] == pytest.approx(0.6)
    assert baseline["balanced_accuracy"] == pytest.approx(1 / 3)
    assert baseline["macro_f1"] == pytest.approx(0.25)


def test_calculate_metrics_rejects_wrong_probability_shape() -> None:
    with pytest.raises(ValueError, match="概率矩阵形状"):
        calculate_metrics(["甲"], np.asarray([[0.5, 0.5]]), ["甲"])
