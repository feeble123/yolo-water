from __future__ import annotations

import json

from water_agent.vision.label_review import build_label_review_manifest


def test_review_manifest_keeps_all_normal_errors_and_caps_other_labels(tmp_path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for file_name in ("normal1.jpg", "normal2.jpg", "pile1.jpg", "pile2.jpg"):
        (images / file_name).write_bytes(b"image")
    oof = tmp_path / "oof.json"
    records = [
        {
            "file": "正常/normal1.jpg",
            "actual": "正常",
            "predicted": "有漂浮物",
            "confidence": 0.9,
            "fold": 0,
            "top2": [],
            "probabilities": {"正常": 0.01},
        },
        {
            "file": "正常/normal2.jpg",
            "actual": "正常",
            "predicted": "有漂浮物",
            "confidence": 0.8,
            "fold": 1,
            "top2": [],
            "probabilities": {"正常": 0.02},
        },
        {
            "file": "乱堆/pile1.jpg",
            "actual": "乱堆",
            "predicted": "有漂浮物",
            "confidence": 0.8,
            "fold": 0,
            "top2": [],
            "probabilities": {"乱堆": 0.01},
        },
        {
            "file": "乱堆/pile2.jpg",
            "actual": "乱堆",
            "predicted": "有漂浮物",
            "confidence": 0.7,
            "fold": 1,
            "top2": [],
            "probabilities": {"乱堆": 0.02},
        },
    ]
    oof.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    report = build_label_review_manifest(
        oof_predictions_path=oof,
        training_images_dir=images,
        output_path=tmp_path / "review.json",
        max_errors_per_label=1,
    )

    assert report["candidate_count"] == 3
    assert [item["file"] for item in report["candidates"]] == [
        "pile1.jpg",
        "normal1.jpg",
        "normal2.jpg",
    ]
