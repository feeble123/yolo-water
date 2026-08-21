from __future__ import annotations

from pathlib import Path

import pytest

from water_agent.vision.classifier import average_probabilities, weight_version


def test_average_probabilities_uses_equal_model_weights() -> None:
    rows = [
        [0.6, 0.1, 0.1, 0.1, 0.05, 0.05],
        [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
    ]

    averaged = average_probabilities(rows)

    assert averaged == pytest.approx([0.4, 0.15, 0.15, 0.15, 0.075, 0.075])


def test_average_probabilities_rejects_incomplete_classes() -> None:
    with pytest.raises(ValueError, match="完整六类"):
        average_probabilities([[0.5, 0.5]])


def test_weight_version_includes_training_run_name() -> None:
    path = Path("artifacts/training/fold1_weighted/weights/best.pt")

    assert weight_version(path) == "fold1_weighted/best.pt"
