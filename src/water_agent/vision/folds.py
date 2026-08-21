from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from water_agent.schemas import SubmissionRecord


def group_ids_from_audit(filenames: list[str], near_groups: list[list[str]]) -> list[str]:
    known = set(filenames)
    mapping: dict[str, str] = {}
    for names in near_groups:
        if len(names) < 2:
            raise ValueError("近重复组至少应包含两个文件")
        group_id = f"near:{min(names)}"
        for name in names:
            if name not in known:
                raise ValueError(f"审计文件包含未知图片: {name}")
            if name in mapping:
                raise ValueError(f"图片出现在多个近重复组: {name}")
            mapping[name] = group_id
    return [mapping.get(name, f"single:{name}") for name in filenames]


def assign_validation_folds(
    labels: list[str],
    groups: list[str],
    n_splits: int,
    seed: int,
) -> list[int]:
    if len(labels) != len(groups):
        raise ValueError("labels与groups长度不一致")
    class_counts = Counter(labels)
    if min(class_counts.values()) < n_splits:
        raise ValueError("最少类别样本数小于折数")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_by_index = [-1] * len(labels)
    features = np.zeros((len(labels), 1), dtype=np.uint8)
    for fold, (_, validation_indices) in enumerate(splitter.split(features, labels, groups)):
        for index in validation_indices:
            fold_by_index[int(index)] = fold
    if any(fold < 0 for fold in fold_by_index):
        raise RuntimeError("存在未分配折的样本")
    group_folds: dict[str, set[int]] = {}
    for group, fold in zip(groups, fold_by_index, strict=True):
        group_folds.setdefault(group, set()).add(fold)
    leaked = [group for group, folds in group_folds.items() if len(folds) > 1]
    if leaked:
        raise RuntimeError(f"近重复组被拆分到不同验证折: {leaked[:3]}")
    return fold_by_index


def _copy_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if os.path.samefile(source, target):
            raise RuntimeError(f"拒绝使用与原图共享文件实体的目标: {target}")
        if source.stat().st_size != target.stat().st_size:
            raise FileExistsError(f"目标已存在且大小不一致: {target}")
        return
    shutil.copy2(source, target)


def prepare_grouped_folds(
    annotations_path: Path,
    images_dir: Path,
    audit_path: Path,
    output_dir: Path,
    n_splits: int = 4,
    seed: int = 20260820,
) -> dict[str, Any]:
    payload = json.loads(annotations_path.read_text(encoding="utf-8-sig"))
    records = [SubmissionRecord.model_validate(item) for item in payload]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    filenames = [record.filename for record in records]
    labels = [record.label.value for record in records]
    groups = group_ids_from_audit(filenames, audit["near_duplicate_groups"])
    validation_folds = assign_validation_folds(labels, groups, n_splits, seed)

    manifest: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold in range(n_splits):
        train_counts: Counter[str] = Counter()
        validation_counts: Counter[str] = Counter()
        for record, group, validation_fold in zip(
            records, groups, validation_folds, strict=True
        ):
            split = "val" if validation_fold == fold else "train"
            source = images_dir / record.filename
            target = output_dir / f"fold_{fold}" / split / record.label.value / record.filename
            _copy_image(source, target)
            (validation_counts if split == "val" else train_counts)[record.label.value] += 1
            manifest.append(
                {
                    "fold": fold,
                    "split": split,
                    "filename": record.filename,
                    "label": record.label.value,
                    "group": group,
                }
            )
        fold_summaries.append(
            {
                "fold": fold,
                "train_count": sum(train_counts.values()),
                "validation_count": sum(validation_counts.values()),
                "train_by_label": dict(sorted(train_counts.items())),
                "validation_by_label": dict(sorted(validation_counts.items())),
            }
        )
    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "n_splits": n_splits,
        "seed": seed,
        "group_count": len(set(groups)),
        "manifest": str(manifest_path),
        "folds": fold_summaries,
    }


def verify_fold_copies(images_dir: Path, output_dir: Path) -> dict[str, int | bool]:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = 0
    shared_file_entities = 0
    size_mismatches = 0
    for item in manifest:
        source = images_dir / item["filename"]
        derived = (
            output_dir
            / f"fold_{item['fold']}"
            / item["split"]
            / item["label"]
            / item["filename"]
        )
        if not derived.exists():
            missing += 1
            continue
        shared_file_entities += int(os.path.samefile(source, derived))
        size_mismatches += int(source.stat().st_size != derived.stat().st_size)
    valid = missing == 0 and shared_file_entities == 0 and size_mismatches == 0
    return {
        "valid": valid,
        "entries": len(manifest),
        "missing": missing,
        "shared_file_entities": shared_file_entities,
        "size_mismatches": size_mismatches,
    }
