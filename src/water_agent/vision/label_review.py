from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from water_agent.labels import WaterLabel


def build_label_review_manifest(
    *,
    oof_predictions_path: Path,
    training_images_dir: Path,
    output_path: Path,
    max_errors_per_label: int = 4,
) -> dict[str, Any]:
    """Create a read-only human review queue from official-training OOF mistakes."""
    if not oof_predictions_path.is_file():
        raise FileNotFoundError(f"OOF预测文件不存在: {oof_predictions_path}")
    if not training_images_dir.is_dir():
        raise FileNotFoundError(f"训练图片目录不存在: {training_images_dir}")
    if max_errors_per_label < 1:
        raise ValueError("每类最大错例数必须为正数")
    records = json.loads(oof_predictions_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError("OOF预测文件必须是记录列表")

    labels = [label.value for label in WaterLabel]
    errors_by_actual: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        actual = str(record["actual"])
        predicted = str(record["predicted"])
        if actual not in labels or predicted not in labels or actual == predicted:
            continue
        file_name = Path(str(record["file"])).name
        image_path = training_images_dir / file_name
        true_probability = float(record["probabilities"][actual])
        errors_by_actual[actual].append(
            {
                "file": file_name,
                "training_image_path": str(image_path.resolve()),
                "official_label": actual,
                "model_prediction": predicted,
                "model_confidence": float(record["confidence"]),
                "official_label_probability": true_probability,
                "fold": int(record["fold"]),
                "top2": record["top2"],
                "review_status": "pending",
                "reviewer_label": None,
                "review_note": None,
            }
        )

    candidates: list[dict[str, Any]] = []
    for label in labels:
        ranked = sorted(
            errors_by_actual[label], key=lambda item: item["official_label_probability"]
        )
        limit = len(ranked) if label == WaterLabel.NORMAL.value else max_errors_per_label
        candidates.extend(ranked[:limit])

    report = {
        "purpose": "仅供人工核验官方训练标签，不会自动修改标签、权重或提交结果。",
        "instructions": [
            "逐张打开training_image_path，并以赛事官方类别定义判断是否支持official_label。",
            "若确认官方标签正确，将review_status改为confirmed，并填写review_note。",
            "若认为需要赛事方澄清，使用needs_official_clarification；不要自行替换官方标签。",
            "只有获得明确人工确认且重新冻结验证集后，才能开展标注修订训练实验。",
        ],
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
