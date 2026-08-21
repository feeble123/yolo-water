from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from water_agent.labels import WaterLabel

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def calculate_metrics(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    label_order: Sequence[str],
) -> dict[str, Any]:
    labels = list(label_order)
    truth = list(y_true)
    scores = np.asarray(probabilities, dtype=np.float64)
    if not truth:
        raise ValueError("评估集不能为空")
    if len(set(labels)) != len(labels) or not labels:
        raise ValueError("label_order必须是非空且无重复的标签列表")
    if scores.shape != (len(truth), len(labels)):
        raise ValueError("概率矩阵形状必须为[样本数, 标签数]")
    unknown = sorted(set(truth) - set(labels))
    if unknown:
        raise ValueError(f"真实标签不在label_order中: {unknown}")
    if not np.isfinite(scores).all():
        raise ValueError("概率矩阵包含非有限值")

    predicted_indices = scores.argmax(axis=1)
    y_pred = [labels[index] for index in predicted_indices]
    precision, recall, class_f1, support = precision_recall_fscore_support(
        truth,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    top_k = min(2, len(labels))
    top_indices = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
    label_to_index = {label: index for index, label in enumerate(labels)}
    top2_correct = [label_to_index[label] in row for label, row in zip(truth, top_indices)]

    return {
        "sample_count": len(truth),
        "label_order": labels,
        "accuracy": float(accuracy_score(truth, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, y_pred)),
        "macro_f1": float(f1_score(truth, y_pred, labels=labels, average="macro")),
        "weighted_f1": float(f1_score(truth, y_pred, labels=labels, average="weighted")),
        "top2_accuracy": float(np.mean(top2_correct)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(truth, y_pred, labels=labels).tolist(),
    }


def calculate_majority_baseline(
    y_true: Sequence[str], label_order: Sequence[str]
) -> dict[str, Any]:
    truth = list(y_true)
    labels = list(label_order)
    if not truth:
        raise ValueError("评估集不能为空")
    majority_label = Counter(truth).most_common(1)[0][0]
    probabilities = np.zeros((len(truth), len(labels)), dtype=np.float64)
    probabilities[:, labels.index(majority_label)] = 1.0
    metrics = calculate_metrics(truth, probabilities, labels)
    metrics["majority_label"] = majority_label
    return metrics


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature必须大于0")
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("概率矩阵必须是至少两类的二维数组")
    logits = np.log(np.clip(scores, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def calculate_calibration_metrics(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    label_order: Sequence[str],
    *,
    bins: int = 10,
) -> dict[str, float]:
    truth = list(y_true)
    labels = list(label_order)
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.shape != (len(truth), len(labels)) or not truth:
        raise ValueError("真实标签与概率矩阵形状不匹配")
    if bins <= 0:
        raise ValueError("bins必须为正数")
    row_sums = scores.sum(axis=1)
    if np.any(scores < 0) or not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("每行概率必须非负且总和为1")

    label_to_index = {label: index for index, label in enumerate(labels)}
    truth_indices = np.asarray([label_to_index[label] for label in truth])
    predicted_indices = scores.argmax(axis=1)
    confidence = scores.max(axis=1)
    correct = predicted_indices == truth_indices
    true_probabilities = scores[np.arange(len(truth)), truth_indices]
    one_hot = np.eye(len(labels), dtype=np.float64)[truth_indices]
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return {
        "nll": float(-np.log(np.clip(true_probabilities, 1e-12, 1.0)).mean()),
        "brier": float(np.square(scores - one_hot).sum(axis=1).mean()),
        "ece": ece,
        "mean_confidence": float(confidence.mean()),
    }


def fit_temperature(
    y_true: Sequence[str], probabilities: np.ndarray, label_order: Sequence[str]
) -> float:
    truth = list(y_true)
    labels = list(label_order)
    label_to_index = {label: index for index, label in enumerate(labels)}
    truth_indices = np.asarray([label_to_index[label] for label in truth])
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.shape != (len(truth), len(labels)) or not truth:
        raise ValueError("真实标签与概率矩阵形状不匹配")

    low, high = -3.0, 3.0
    best_log_temperature = 0.0
    for _ in range(5):
        candidates = np.linspace(low, high, 61)
        losses = []
        for candidate in candidates:
            calibrated = apply_temperature(scores, float(np.exp(candidate)))
            losses.append(
                float(
                    -np.log(
                        np.clip(calibrated[np.arange(len(truth)), truth_indices], 1e-12, 1.0)
                    ).mean()
                )
            )
        best_index = int(np.argmin(losses))
        best_log_temperature = float(candidates[best_index])
        step = float(candidates[1] - candidates[0])
        low, high = best_log_temperature - step, best_log_temperature + step
    return float(np.exp(best_log_temperature))


def _configure_runtime(output_dir: Path) -> None:
    config_root = output_dir.parent
    ultralytics_dir = config_root / "ultralytics_config"
    matplotlib_dir = config_root / "matplotlib_config"
    ultralytics_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_dir.resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir.resolve()))
    if os.name == "nt" and "WINDIR" not in os.environ:
        os.environ["WINDIR"] = os.environ.get("SystemRoot", r"C:\Windows")


def _collect_validation_images(data_dir: Path, labels: Sequence[str]) -> list[Path]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"验证集目录不存在: {data_dir}")
    images: list[Path] = []
    for label in labels:
        class_dir = data_dir / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"验证集缺少类别目录: {class_dir}")
        images.extend(
            path
            for path in sorted(class_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    if not images:
        raise ValueError("验证集没有可评估图片")
    return images


def _write_confusion_csv(path: Path, labels: Sequence[str], matrix: Sequence[Sequence[int]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def evaluate_model(
    *,
    weights: Path,
    data_dir: Path,
    output_dir: Path,
    image_size: int,
    batch: int,
    device: str,
) -> dict[str, Any]:
    if not weights.is_file():
        raise FileNotFoundError(f"模型权重不存在: {weights}")
    if image_size <= 0 or batch <= 0:
        raise ValueError("image_size与batch必须为正数")

    labels = [label.value for label in WaterLabel]
    image_paths = _collect_validation_images(data_dir, labels)
    output_dir.mkdir(parents=True, exist_ok=False)
    _configure_runtime(output_dir)

    from ultralytics import YOLO

    model = YOLO(str(weights.resolve()))
    started = time.perf_counter()
    results = model.predict(
        source=[str(path.resolve()) for path in image_paths],
        imgsz=image_size,
        batch=batch,
        device=device,
        save=False,
        verbose=False,
        stream=True,
    )

    y_true: list[str] = []
    score_rows: list[list[float]] = []
    predictions: list[dict[str, Any]] = []
    for result in results:
        if result.probs is None:
            raise RuntimeError("加载的Ultralytics权重不是分类模型")
        actual = Path(result.path).parent.name
        model_scores = result.probs.data.detach().cpu().tolist()
        by_label = {str(result.names[index]): float(value) for index, value in enumerate(model_scores)}
        missing = sorted(set(labels) - set(by_label))
        if missing:
            raise ValueError(f"模型类别与官方类别不一致，缺少: {missing}")
        row = [by_label[label] for label in labels]
        predicted = labels[int(np.argmax(row))]
        ranking = np.argsort(row)[::-1]
        y_true.append(actual)
        score_rows.append(row)
        predictions.append(
            {
                "file": str(Path(result.path).resolve().relative_to(data_dir.resolve())),
                "actual": actual,
                "predicted": predicted,
                "confidence": float(max(row)),
                "top2": [
                    {"label": labels[index], "probability": float(row[index])}
                    for index in ranking[:2]
                ],
                "probabilities": {label: float(value) for label, value in zip(labels, row, strict=True)},
            }
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    if len(y_true) != len(image_paths):
        raise RuntimeError(f"推理结果数量不匹配: 期望{len(image_paths)}，实际{len(y_true)}")

    metrics = calculate_metrics(y_true, np.asarray(score_rows), labels)
    report = {
        "model": str(weights.resolve()),
        "dataset": str(data_dir.resolve()),
        "image_size": image_size,
        "batch": batch,
        "device": device,
        "elapsed_ms": elapsed_ms,
        "mean_latency_ms": elapsed_ms / len(y_true),
        "metrics": metrics,
        "majority_baseline": calculate_majority_baseline(y_true, labels),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_confusion_csv(
        output_dir / "confusion_matrix.csv", labels, metrics["confusion_matrix"]
    )
    return {
        "output_dir": str(output_dir.resolve()),
        "sample_count": metrics["sample_count"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "majority_accuracy": report["majority_baseline"]["accuracy"],
        "mean_latency_ms": report["mean_latency_ms"],
    }
