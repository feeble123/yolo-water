from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from water_agent.labels import WaterLabel
from water_agent.vision.evaluate import (
    apply_temperature,
    calculate_calibration_metrics,
    calculate_majority_baseline,
    calculate_metrics,
    fit_temperature,
)


def _write_confusion_csv(path: Path, labels: Sequence[str], matrix: Sequence[Sequence[int]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def aggregate_oof(evaluation_dirs: Sequence[Path], output_dir: Path) -> dict[str, Any]:
    if len(evaluation_dirs) < 2:
        raise ValueError("OOF汇总至少需要两个评估目录")
    labels = [label.value for label in WaterLabel]
    all_records: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    seen_files: set[str] = set()

    for fold_index, evaluation_dir in enumerate(evaluation_dirs):
        metrics_path = evaluation_dir / "metrics.json"
        predictions_path = evaluation_dir / "predictions.json"
        if not metrics_path.is_file() or not predictions_path.is_file():
            raise FileNotFoundError(f"评估目录缺少metrics.json或predictions.json: {evaluation_dir}")
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        records = json.loads(predictions_path.read_text(encoding="utf-8"))
        fold_summaries.append(
            {
                "fold": fold_index,
                "evaluation_dir": str(evaluation_dir.resolve()),
                "sample_count": report["metrics"]["sample_count"],
                "accuracy": report["metrics"]["accuracy"],
                "balanced_accuracy": report["metrics"]["balanced_accuracy"],
                "macro_f1": report["metrics"]["macro_f1"],
            }
        )
        for record in records:
            file_key = str(record["file"])
            if file_key in seen_files:
                raise ValueError(f"OOF验证样本重复出现: {file_key}")
            seen_files.add(file_key)
            record["fold"] = fold_index
            all_records.append(record)

    y_true = [str(record["actual"]) for record in all_records]
    probabilities = np.asarray(
        [[float(record["probabilities"][label]) for label in labels] for record in all_records]
    )
    temperature = fit_temperature(y_true, probabilities, labels)
    calibrated = apply_temperature(probabilities, temperature)
    metrics = calculate_metrics(y_true, probabilities, labels)
    calibrated_metrics = calculate_metrics(y_true, calibrated, labels)
    calibration_before = calculate_calibration_metrics(y_true, probabilities, labels)
    calibration_after = calculate_calibration_metrics(y_true, calibrated, labels)

    for record, calibrated_row in zip(all_records, calibrated, strict=True):
        predicted_index = int(np.argmax(calibrated_row))
        record["calibrated"] = {
            "predicted": labels[predicted_index],
            "confidence": float(calibrated_row[predicted_index]),
            "probabilities": {
                label: float(value)
                for label, value in zip(labels, calibrated_row.tolist(), strict=True)
            },
        }

    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "fold_count": len(evaluation_dirs),
        "unique_sample_count": len(seen_files),
        "folds": fold_summaries,
        "metrics": metrics,
        "majority_baseline": calculate_majority_baseline(y_true, labels),
        "calibration": {
            "temperature": temperature,
            "before": calibration_before,
            "after": calibration_after,
            "classification_metrics_after": calibrated_metrics,
        },
    }
    (output_dir / "oof_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "oof_predictions.json").write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_confusion_csv(output_dir / "oof_confusion_matrix.csv", labels, metrics["confusion_matrix"])
    return {
        "output_dir": str(output_dir.resolve()),
        "fold_count": len(evaluation_dirs),
        "sample_count": len(seen_files),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "temperature": temperature,
        "ece_before": calibration_before["ece"],
        "ece_after": calibration_after["ece"],
    }
