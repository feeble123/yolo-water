from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from water_agent.labels import WaterLabel
from water_agent.schemas import SubmissionRecord
from water_agent.vision.classifier import average_probabilities
from water_agent.vision.evaluate import IMAGE_SUFFIXES, apply_temperature


def _configure_runtime(output: Path) -> None:
    config_root = output.parent / "batch_runtime"
    ultralytics_dir = config_root / "ultralytics_config"
    matplotlib_dir = config_root / "matplotlib_config"
    ultralytics_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_dir.resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir.resolve()))
    if os.name == "nt" and "WINDIR" not in os.environ:
        os.environ["WINDIR"] = os.environ.get("SystemRoot", r"C:\Windows")


def predict_batch(
    *,
    input_dir: Path,
    weights: Sequence[Path],
    output: Path,
    image_size: int,
    batch: int,
    device: str,
    temperature: float = 1.0,
) -> dict[str, Any]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入图片目录不存在: {input_dir}")
    if len(weights) < 1:
        raise ValueError("至少需要一个模型权重")
    missing_weights = [str(path) for path in weights if not path.is_file()]
    if missing_weights:
        raise FileNotFoundError(f"模型权重不存在: {missing_weights}")
    if output.exists() or output.with_name(f"{output.stem}.audit.json").exists():
        raise FileExistsError("输出JSON或其审计文件已存在，请更换路径以避免覆盖")
    if image_size <= 0 or batch <= 0 or temperature <= 0:
        raise ValueError("image_size、batch与temperature必须为正数")

    image_paths = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError("输入目录没有支持的图片")
    basenames = [path.name for path in image_paths]
    if len(basenames) != len(set(basenames)):
        raise ValueError("输入目录存在同名图片，赛事filename字段会冲突")

    output.parent.mkdir(parents=True, exist_ok=True)
    _configure_runtime(output)
    from ultralytics import YOLO

    labels = [label.value for label in WaterLabel]
    path_keys = [str(path.resolve()).casefold() for path in image_paths]
    accumulated: dict[str, list[list[float]]] = {key: [] for key in path_keys}
    started = time.perf_counter()
    for weight_path in weights:
        model = YOLO(str(weight_path.resolve()))
        results = model.predict(
            source=[str(path.resolve()) for path in image_paths],
            imgsz=image_size,
            batch=batch,
            device=device,
            save=False,
            verbose=False,
            stream=True,
        )
        result_count = 0
        for result in results:
            if result.probs is None:
                raise RuntimeError("加载的Ultralytics权重不是分类模型")
            key = str(Path(result.path).resolve()).casefold()
            if key not in accumulated:
                raise RuntimeError(f"模型返回了输入目录之外的结果: {result.path}")
            values = result.probs.data.detach().cpu().tolist()
            by_label = {
                str(result.names[index]): float(value) for index, value in enumerate(values)
            }
            if set(by_label) != set(labels):
                raise ValueError(f"模型类别与官方六类不一致: {weight_path}")
            accumulated[key].append([by_label[label] for label in labels])
            result_count += 1
        if result_count != len(image_paths):
            raise RuntimeError(
                f"模型推理结果数量不匹配: {weight_path}，期望{len(image_paths)}，实际{result_count}"
            )
        del model

    submission: list[dict[str, str]] = []
    audit_records: list[dict[str, Any]] = []
    for path, key in zip(image_paths, path_keys, strict=True):
        averaged = np.asarray([average_probabilities(accumulated[key])])
        probabilities = apply_temperature(averaged, temperature)[0]
        predicted_index = int(np.argmax(probabilities))
        with Image.open(path) as image:
            width, height = image.size
        record = SubmissionRecord(
            filename=path.name,
            width=str(width),
            height=str(height),
            label=WaterLabel(labels[predicted_index]),
        )
        submission.append(record.model_dump(mode="json"))
        audit_records.append(
            {
                "filename": path.name,
                "label": labels[predicted_index],
                "confidence": float(probabilities[predicted_index]),
                "probabilities": {
                    label: float(value)
                    for label, value in zip(labels, probabilities.tolist(), strict=True)
                },
            }
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    output.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path = output.with_name(f"{output.stem}.audit.json")
    audit_path.write_text(
        json.dumps(
            {
                "input_dir": str(input_dir.resolve()),
                "weights": [str(path.resolve()) for path in weights],
                "temperature": temperature,
                "image_size": image_size,
                "sample_count": len(image_paths),
                "elapsed_ms": elapsed_ms,
                "predictions": audit_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "output": str(output.resolve()),
        "audit": str(audit_path.resolve()),
        "sample_count": len(image_paths),
        "model_count": len(weights),
        "elapsed_ms": elapsed_ms,
    }
