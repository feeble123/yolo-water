from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from water_agent.schemas import SubmissionRecord


def difference_hash(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _groups_from_pairs(items: list[dict[str, Any]], max_distance: int) -> list[list[str]]:
    groups: list[list[str]] = []
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        by_label[item["label"]].append(index)
    disjoint = _DisjointSet(len(items))
    for indices in by_label.values():
        for position, left in enumerate(indices):
            for right in indices[position + 1 :]:
                if (items[left]["dhash"] ^ items[right]["dhash"]).bit_count() <= max_distance:
                    disjoint.union(left, right)
    components: dict[int, list[str]] = defaultdict(list)
    for index, item in enumerate(items):
        components[disjoint.find(index)].append(item["filename"])
    groups.extend(sorted(names) for names in components.values() if len(names) > 1)
    return sorted(groups, key=lambda names: (-len(names), names[0]))


def audit_training_data(
    annotations_path: Path,
    images_dir: Path,
    near_duplicate_distance: int = 6,
) -> dict[str, Any]:
    if not 0 <= near_duplicate_distance <= 64:
        raise ValueError("near_duplicate_distance必须在0到64之间")
    raw = json.loads(annotations_path.read_text(encoding="utf-8-sig"))
    records = [SubmissionRecord.model_validate(item) for item in raw]
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    exact: dict[str, list[str]] = defaultdict(list)
    size_mismatches: list[str] = []
    for record in records:
        if record.filename in seen:
            raise ValueError(f"标签文件包含重复文件名: {record.filename}")
        seen.add(record.filename)
        path = images_dir / record.filename
        data = path.read_bytes()
        exact[hashlib.sha256(data).hexdigest()].append(record.filename)
        with Image.open(path) as image:
            if image.size != (int(record.width), int(record.height)):
                size_mismatches.append(record.filename)
            hash_value = difference_hash(image)
        items.append(
            {"filename": record.filename, "label": record.label.value, "dhash": hash_value}
        )
    exact_groups = sorted(
        (sorted(names) for names in exact.values() if len(names) > 1),
        key=lambda names: (-len(names), names[0]),
    )
    near_groups = _groups_from_pairs(items, near_duplicate_distance)
    label_counts = Counter(record.label.value for record in records)
    return {
        "record_count": len(records),
        "label_counts": dict(sorted(label_counts.items())),
        "size_mismatches": size_mismatches,
        "exact_duplicate_groups": exact_groups,
        "near_duplicate_distance": near_duplicate_distance,
        "near_duplicate_groups": near_groups,
        "near_duplicate_images": sum(len(group) for group in near_groups),
        "note": "近重复由同标签64位dHash汉明距离聚类，仅用于切分风险审计，不代表语义完全相同。",
    }


def write_audit(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
