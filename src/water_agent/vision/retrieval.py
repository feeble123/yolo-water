from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from water_agent.labels import WaterLabel
from water_agent.vision.evaluate import IMAGE_SUFFIXES, calculate_metrics


def _collect_images(root: Path, labels: Sequence[str]) -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    targets: list[str] = []
    for label in labels:
        directory = root / label
        if not directory.is_dir():
            raise FileNotFoundError(f"数据目录缺少类别: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(path)
                targets.append(label)
    if not paths:
        raise ValueError(f"数据目录没有图片: {root}")
    return paths, targets


def _chunks(items: Sequence[Path], size: int) -> Iterable[Sequence[Path]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _extract_embeddings(model: Any, paths: Sequence[Path], *, image_size: int, batch: int, device: str) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    for batch_paths in _chunks(paths, batch):
        outputs = model.embed(
            [str(path.resolve()) for path in batch_paths],
            imgsz=image_size,
            batch=batch,
            device=device,
            verbose=False,
        )
        embeddings.extend(output.detach().cpu().numpy().astype(np.float64) for output in outputs)
    matrix = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def _predict_probabilities(
    model: Any, paths: Sequence[Path], labels: Sequence[str], *, image_size: int, batch: int, device: str
) -> np.ndarray:
    rows: list[list[float]] = []
    for batch_paths in _chunks(paths, batch):
        results = model.predict(
            source=[str(path.resolve()) for path in batch_paths],
            imgsz=image_size,
            batch=batch,
            device=device,
            verbose=False,
            save=False,
            stream=True,
        )
        for result in results:
            if result.probs is None:
                raise RuntimeError("加载的权重不是分类模型")
            scores_by_label = {
                str(result.names[index]): float(value)
                for index, value in enumerate(result.probs.data.detach().cpu().tolist())
            }
            if set(scores_by_label) != set(labels):
                raise ValueError("模型类别与官方类别不一致")
            rows.append([scores_by_label[label] for label in labels])
    return np.asarray(rows, dtype=np.float64)


def knn_class_probabilities(
    query_embeddings: np.ndarray,
    bank_embeddings: np.ndarray,
    bank_labels: Sequence[str],
    label_order: Sequence[str],
    *,
    k: int,
) -> np.ndarray:
    """Return cosine-similarity weighted KNN class probabilities."""
    query = np.asarray(query_embeddings, dtype=np.float64)
    bank = np.asarray(bank_embeddings, dtype=np.float64)
    labels = list(label_order)
    if query.ndim != 2 or bank.ndim != 2 or query.shape[1] != bank.shape[1]:
        raise ValueError("查询与记忆库embedding维度不一致")
    if len(bank) != len(bank_labels) or not 1 <= k <= len(bank):
        raise ValueError("KNN记忆库或k参数不合法")
    similarities = query @ bank.T
    nearest = np.argpartition(similarities, -k, axis=1)[:, -k:]
    output = np.zeros((len(query), len(labels)), dtype=np.float64)
    label_to_index = {label: index for index, label in enumerate(labels)}
    for row_index, neighbors in enumerate(nearest):
        weights = np.maximum(similarities[row_index, neighbors], 0.0) + 1e-6
        for neighbor_index, weight in zip(neighbors, weights, strict=True):
            output[row_index, label_to_index[bank_labels[int(neighbor_index)]]] += float(weight)
    return output / output.sum(axis=1, keepdims=True)


def fuse_probabilities(classifier: np.ndarray, retrieval: np.ndarray, *, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("融合权重alpha必须在[0, 1]之间")
    first = np.asarray(classifier, dtype=np.float64)
    second = np.asarray(retrieval, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("分类器与检索概率矩阵形状不一致")
    return (1.0 - alpha) * first + alpha * second


def screen_retrieval_hybrid(
    *,
    weights: Path,
    data_dir: Path,
    output_path: Path,
    image_size: int,
    batch: int,
    device: str,
) -> dict[str, Any]:
    """Screen a train-only KNN visual-memory hybrid on one isolated validation fold."""
    if not weights.is_file():
        raise FileNotFoundError(f"模型权重不存在: {weights}")
    if batch <= 0 or image_size <= 0:
        raise ValueError("batch和image_size必须为正数")
    labels = [label.value for label in WaterLabel]
    train_paths, train_targets = _collect_images(data_dir / "train", labels)
    val_paths, val_targets = _collect_images(data_dir / "val", labels)
    from ultralytics import YOLO

    model = YOLO(str(weights.resolve()))
    started = time.perf_counter()
    train_embeddings = _extract_embeddings(model, train_paths, image_size=image_size, batch=batch, device=device)
    val_embeddings = _extract_embeddings(model, val_paths, image_size=image_size, batch=batch, device=device)
    classifier_scores = _predict_probabilities(
        model, val_paths, labels, image_size=image_size, batch=batch, device=device
    )
    baseline = calculate_metrics(val_targets, classifier_scores, labels)
    candidates: list[dict[str, Any]] = []
    best_metrics = baseline
    best_config = {"mode": "classifier", "k": None, "alpha": 0.0}
    for k in (1, 3, 5, 9):
        retrieval_scores = knn_class_probabilities(
            val_embeddings, train_embeddings, train_targets, labels, k=k
        )
        for alpha in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
            scores = fuse_probabilities(classifier_scores, retrieval_scores, alpha=alpha)
            metrics = calculate_metrics(val_targets, scores, labels)
            candidate = {"mode": "hybrid", "k": k, "alpha": alpha, "metrics": metrics}
            candidates.append(candidate)
            key = (metrics["macro_f1"], metrics["balanced_accuracy"], metrics["accuracy"])
            best_key = (
                best_metrics["macro_f1"],
                best_metrics["balanced_accuracy"],
                best_metrics["accuracy"],
            )
            if key > best_key:
                best_metrics = metrics
                best_config = {"mode": "hybrid", "k": k, "alpha": alpha}
    report = {
        "weights": str(weights.resolve()),
        "data_dir": str(data_dir.resolve()),
        "sample_counts": {"train": len(train_paths), "val": len(val_paths)},
        "embedding_dimension": int(train_embeddings.shape[1]),
        "baseline_metrics": baseline,
        "best_config": best_config,
        "best_metrics": best_metrics,
        "improved_over_classifier": best_config["mode"] == "hybrid",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "safety_note": "仅以当前fold训练目录建立检索记忆库；验证图片从未进入记忆库。单fold筛选不构成上线证据，需四折复验。",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
