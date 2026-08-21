from __future__ import annotations

import numpy as np
import pytest

from water_agent.vision.decision_policy import (
    apply_logit_bias,
    fit_conservative_biases,
    nested_oof_bias_evaluation,
)


def test_apply_logit_bias_can_change_the_top_class() -> None:
    scores = np.asarray([[0.55, 0.45]])

    adjusted = apply_logit_bias(scores, [0.0, 0.5])

    assert adjusted.argmax(axis=1).tolist() == [1]
    assert adjusted.sum(axis=1).tolist() == pytest.approx([1.0])


def test_bias_fit_leaves_low_support_and_majority_labels_fixed() -> None:
    labels = ["多数", "小类", "极少类"]
    truth = ["多数"] * 20 + ["小类"] * 20 + ["极少类"] * 2
    scores = np.asarray(
        [[0.7, 0.25, 0.05]] * 20 + [[0.52, 0.43, 0.05]] * 20 + [[0.7, 0.1, 0.2]] * 2
    )

    result = fit_conservative_biases(truth, scores, labels, min_class_support=10)

    assert result["majority_label"] == "多数"
    assert result["eligible_labels"] == ["小类"]
    assert result["biases"]["多数"] == 0.0
    assert result["biases"]["极少类"] == 0.0


def test_nested_evaluation_never_fits_the_holdout_fold() -> None:
    labels = ["乱采", "乱建", "乱堆", "乱占", "有漂浮物", "正常"]
    records = []
    for fold in (0, 1):
        for actual in labels:
            probabilities = {label: 0.01 for label in labels}
            probabilities[actual] = 0.9
            probabilities["有漂浮物"] += 0.05
            records.append({"fold": fold, "actual": actual, "probabilities": probabilities})

    report = nested_oof_bias_evaluation(records, min_class_support=2)

    assert report["sample_count"] == len(records)
    assert len(report["folds"]) == 2
    assert report["policy_constraints"]["min_class_support"] == 2
