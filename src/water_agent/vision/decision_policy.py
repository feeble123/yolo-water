from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from water_agent.labels import WaterLabel
from water_agent.vision.evaluate import calculate_metrics


def apply_logit_bias(probabilities: np.ndarray, biases: Sequence[float]) -> np.ndarray:
    """Apply fixed class biases in log-probability space and renormalize."""
    scores = np.asarray(probabilities, dtype=np.float64)
    offsets = np.asarray(biases, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(offsets):
        raise ValueError("概率矩阵与类别偏置维度不一致")
    if not np.isfinite(scores).all() or not np.isfinite(offsets).all():
        raise ValueError("概率或偏置包含非有限值")
    if np.any(scores < 0.0):
        raise ValueError("概率不能为负数")
    logits = np.log(np.clip(scores, 1e-12, 1.0)) + offsets
    logits -= logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _metric_key(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["macro_f1"]),
        float(metrics["balanced_accuracy"]),
        float(metrics["accuracy"]),
    )


def fit_conservative_biases(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    label_order: Sequence[str],
    *,
    min_class_support: int = 20,
    max_bias: float = 0.75,
    grid_size: int = 13,
    passes: int = 3,
) -> dict[str, Any]:
    """Fit modest class biases only for adequately represented non-majority classes."""
    labels = list(label_order)
    truth = list(y_true)
    scores = np.asarray(probabilities, dtype=np.float64)
    if len(truth) < 2 or scores.shape != (len(truth), len(labels)):
        raise ValueError("真实标签与概率矩阵形状不匹配")
    if min_class_support < 1 or max_bias <= 0.0 or grid_size < 3 or passes < 1:
        raise ValueError("决策偏置搜索参数不合法")
    counts = Counter(truth)
    majority_label = counts.most_common(1)[0][0]
    eligible = [
        label
        for label in labels
        if label != majority_label and counts[label] >= min_class_support
    ]
    candidates = np.linspace(-max_bias, max_bias, grid_size)
    biases = np.zeros(len(labels), dtype=np.float64)
    best_metrics = calculate_metrics(truth, apply_logit_bias(scores, biases), labels)

    for _ in range(passes):
        changed = False
        for label in eligible:
            index = labels.index(label)
            candidate_biases = []
            candidate_metrics = []
            for value in candidates:
                proposal = biases.copy()
                proposal[index] = float(value)
                candidate_biases.append(proposal)
                candidate_metrics.append(
                    calculate_metrics(truth, apply_logit_bias(scores, proposal), labels)
                )
            choice = max(range(len(candidates)), key=lambda item: _metric_key(candidate_metrics[item]))
            if _metric_key(candidate_metrics[choice]) > _metric_key(best_metrics):
                biases = candidate_biases[choice]
                best_metrics = candidate_metrics[choice]
                changed = True
        if not changed:
            break

    return {
        "biases": {label: float(value) for label, value in zip(labels, biases, strict=True)},
        "majority_label": majority_label,
        "eligible_labels": eligible,
        "min_class_support": min_class_support,
        "max_bias": max_bias,
        "metrics": best_metrics,
    }


def nested_oof_bias_evaluation(
    records: Sequence[Mapping[str, Any]],
    *,
    min_class_support: int = 20,
    max_bias: float = 0.75,
) -> dict[str, Any]:
    """Evaluate class-bias calibration with each OOF fold held out from bias fitting."""
    if len(records) < 2:
        raise ValueError("OOF记录数量不足")
    labels = [label.value for label in WaterLabel]
    folds = sorted({int(record["fold"]) for record in records})
    if len(folds) < 2:
        raise ValueError("OOF记录至少需要两个fold")
    truth = [str(record["actual"]) for record in records]
    raw_scores = np.asarray(
        [[float(record["probabilities"][label]) for label in labels] for record in records],
        dtype=np.float64,
    )
    transformed = np.zeros_like(raw_scores)
    reports: list[dict[str, Any]] = []
    record_folds = np.asarray([int(record["fold"]) for record in records])

    for holdout_fold in folds:
        train_mask = record_folds != holdout_fold
        test_mask = ~train_mask
        fit = fit_conservative_biases(
            [label for label, selected in zip(truth, train_mask, strict=True) if selected],
            raw_scores[train_mask],
            labels,
            min_class_support=min_class_support,
            max_bias=max_bias,
        )
        biases = [fit["biases"][label] for label in labels]
        transformed[test_mask] = apply_logit_bias(raw_scores[test_mask], biases)
        reports.append(
            {
                "holdout_fold": holdout_fold,
                "fit": fit,
                "raw_metrics": calculate_metrics(
                    [label for label, selected in zip(truth, test_mask, strict=True) if selected],
                    raw_scores[test_mask],
                    labels,
                ),
                "biased_metrics": calculate_metrics(
                    [label for label, selected in zip(truth, test_mask, strict=True) if selected],
                    transformed[test_mask],
                    labels,
                ),
            }
        )

    return {
        "label_order": labels,
        "sample_count": len(records),
        "raw_metrics": calculate_metrics(truth, raw_scores, labels),
        "nested_biased_metrics": calculate_metrics(truth, transformed, labels),
        "folds": reports,
        "policy_constraints": {
            "min_class_support": min_class_support,
            "max_bias": max_bias,
            "note": "正常与乱建样本不足，不参与偏置拟合；每个holdout fold均未参与其偏置选择。",
        },
    }


def evaluate_oof_decision_policy(
    *,
    predictions_path: Path,
    output_path: Path,
    min_class_support: int = 20,
    max_bias: float = 0.75,
) -> dict[str, Any]:
    if not predictions_path.is_file():
        raise FileNotFoundError(f"OOF预测文件不存在: {predictions_path}")
    records = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError("OOF预测文件必须是记录列表")
    report = nested_oof_bias_evaluation(
        records,
        min_class_support=min_class_support,
        max_bias=max_bias,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
