from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from water_agent.labels import WaterLabel
from water_agent.schemas import ClassificationResult, ClassScore
from water_agent.vision.evaluate import apply_temperature

CLASS_RISK_REASONS = {
    WaterLabel.ILLEGAL_BUILDING: "乱建类验证样本仅4张且OOF召回率为50%",
    WaterLabel.NORMAL: "正常类验证样本仅7张且OOF召回率为0%",
}


class VisionClassifier(Protocol):
    def classify(self, image_path: Path) -> ClassificationResult: ...


def weight_version(weights: Path) -> str:
    run_name = weights.parents[1].name if len(weights.parents) > 1 else weights.parent.name
    return f"{run_name}/{weights.name}"


def _build_result(
    probabilities: dict[WaterLabel, float],
    *,
    threshold: float,
    model_version: str,
    latency_ms: float,
    force_review_labels: set[WaterLabel] | None = None,
) -> ClassificationResult:
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    top_k = [ClassScore(label=label, probability=probability) for label, probability in ranked[:3]]
    label, confidence = ranked[0]
    reasons: list[str] = []
    if confidence < threshold:
        reasons.append("最高置信度低于人工复核阈值")
    if force_review_labels and label in force_review_labels:
        reasons.append(CLASS_RISK_REASONS.get(label, "该类别验证证据不足或召回不稳定"))
    requires_review = bool(reasons)
    reason = "；".join(reasons) if reasons else None
    return ClassificationResult(
        label=label,
        confidence=confidence,
        top_k=top_k,
        requires_review=requires_review,
        review_reason=reason,
        model_version=model_version,
        latency_ms=latency_ms,
    )


class FakeClassifier:
    """可复现的架构测试替身；输出不得用于比赛。"""

    def __init__(self, review_threshold: float = 0.65) -> None:
        self.review_threshold = review_threshold

    def classify(self, image_path: Path) -> ClassificationResult:
        started = time.perf_counter()
        digest = hashlib.sha256(image_path.read_bytes()).digest()
        labels = list(WaterLabel)
        raw = [float(digest[index] + 1) for index in range(len(labels))]
        total = sum(raw)
        probabilities = {label: value / total for label, value in zip(labels, raw, strict=True)}
        return _build_result(
            probabilities,
            threshold=self.review_threshold,
            model_version="fake-architecture-test-only",
            latency_ms=(time.perf_counter() - started) * 1000,
        )


class UltralyticsClassifier:
    def __init__(
        self,
        weights: Path,
        review_threshold: float = 0.65,
        force_review_labels: set[WaterLabel] | None = None,
        image_size: int = 320,
        device: str = "0",
    ) -> None:
        if not weights.is_file():
            raise FileNotFoundError(f"视觉模型权重不存在: {weights}")
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.review_threshold = review_threshold
        self.force_review_labels = force_review_labels or set()
        self.image_size = image_size
        self.device = device
        self.model_version = weight_version(weights)

    def classify(self, image_path: Path) -> ClassificationResult:
        started = time.perf_counter()
        prediction = self.model.predict(
            source=str(image_path), imgsz=self.image_size, device=self.device, verbose=False
        )[0]
        probs = prediction.probs
        if probs is None:
            raise RuntimeError("加载的Ultralytics权重不是分类模型")
        names = prediction.names
        values = probs.data.detach().cpu().tolist()
        probabilities: dict[WaterLabel, float] = {}
        for index, value in enumerate(values):
            probabilities[WaterLabel(names[index])] = float(value)
        return _build_result(
            probabilities,
            threshold=self.review_threshold,
            model_version=self.model_version,
            latency_ms=(time.perf_counter() - started) * 1000,
            force_review_labels=self.force_review_labels,
        )


def average_probabilities(probability_rows: list[list[float]]) -> list[float]:
    if not probability_rows:
        raise ValueError("至少需要一个模型的概率")
    scores = np.asarray(probability_rows, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(WaterLabel):
        raise ValueError("每个模型必须输出完整六类概率")
    if np.any(scores < 0) or not np.allclose(scores.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("每个模型的概率必须非负且总和为1")
    return scores.mean(axis=0).tolist()


class UltralyticsEnsembleClassifier:
    def __init__(
        self,
        weights: list[Path],
        review_threshold: float = 0.65,
        temperature: float = 1.0,
        force_review_labels: set[WaterLabel] | None = None,
        image_size: int = 320,
        device: str = "0",
    ) -> None:
        if len(weights) < 2:
            raise ValueError("集成分类器至少需要两个权重")
        missing = [str(path) for path in weights if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"视觉模型权重不存在: {missing}")
        if temperature <= 0:
            raise ValueError("temperature必须大于0")
        from ultralytics import YOLO

        self.models = [YOLO(str(path)) for path in weights]
        self.review_threshold = review_threshold
        self.temperature = temperature
        self.force_review_labels = force_review_labels or set()
        self.image_size = image_size
        self.device = device
        self.model_version = "ensemble:" + "+".join(weight_version(path) for path in weights)

    def classify(self, image_path: Path) -> ClassificationResult:
        started = time.perf_counter()
        labels = list(WaterLabel)
        rows: list[list[float]] = []
        for model in self.models:
            prediction = model.predict(
                source=str(image_path),
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )[0]
            if prediction.probs is None:
                raise RuntimeError("加载的Ultralytics权重不是分类模型")
            values = prediction.probs.data.detach().cpu().tolist()
            by_label = {
                WaterLabel(prediction.names[index]): float(value)
                for index, value in enumerate(values)
            }
            rows.append([by_label[label] for label in labels])
        averaged = np.asarray([average_probabilities(rows)])
        calibrated = apply_temperature(averaged, self.temperature)[0].tolist()
        return _build_result(
            dict(zip(labels, calibrated, strict=True)),
            threshold=self.review_threshold,
            model_version=self.model_version,
            latency_ms=(time.perf_counter() - started) * 1000,
            force_review_labels=self.force_review_labels,
        )
