from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from water_agent.schemas import SubmissionRecord, SubmissionValidation


def validate_submission(path: Path) -> SubmissionValidation:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return SubmissionValidation(valid=False, record_count=0, errors=[f"无法读取JSON: {exc}"])
    if not isinstance(payload, list):
        return SubmissionValidation(valid=False, record_count=0, errors=["顶层结构必须是数组"])

    filenames: set[str] = set()
    for index, item in enumerate(payload):
        try:
            record = SubmissionRecord.model_validate(item)
        except ValidationError as exc:
            errors.append(f"第{index + 1}条记录不合法: {exc.errors(include_url=False)}")
            continue
        if record.filename in filenames:
            errors.append(f"文件名重复: {record.filename}")
        filenames.add(record.filename)
    return SubmissionValidation(valid=not errors, record_count=len(payload), errors=errors)

