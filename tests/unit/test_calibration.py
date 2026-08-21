from __future__ import annotations

import numpy as np
import pytest

from water_agent.vision.evaluate import (
    apply_temperature,
    calculate_calibration_metrics,
    fit_temperature,
)


def test_temperature_one_preserves_probabilities() -> None:
    probabilities = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])

    assert apply_temperature(probabilities, 1.0) == pytest.approx(probabilities)


def test_fitted_temperature_does_not_increase_nll() -> None:
    labels = ["甲", "乙"]
    truth = ["甲", "甲", "乙", "乙"]
    overconfident = np.asarray([[0.99, 0.01], [0.99, 0.01], [0.99, 0.01], [0.01, 0.99]])

    temperature = fit_temperature(truth, overconfident, labels)
    calibrated = apply_temperature(overconfident, temperature)
    before = calculate_calibration_metrics(truth, overconfident, labels)
    after = calculate_calibration_metrics(truth, calibrated, labels)

    assert temperature > 1.0
    assert after["nll"] <= before["nll"]
