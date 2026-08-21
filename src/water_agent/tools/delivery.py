from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from water_agent.tools.submission import validate_submission

ALLOWED_DESIGN_SUFFIXES = {".docx", ".pdf", ".md"}
FORBIDDEN_NAMES = {".env", "api key.txt", "apikey.txt", "api_key.txt"}
ALLOWED_TOP_LEVEL = {"code", "design", "result"}


class DeliveryValidation(BaseModel):
    valid: bool
    errors: list[str]
    code_file_count: int = Field(ge=0)
    design_file_count: int = Field(ge=0)
    result_record_count: int = Field(ge=0)


def validate_delivery(root: Path) -> DeliveryValidation:
    errors: list[str] = []
    code_dir = root / "code"
    design_dir = root / "design"
    result_path = root / "result" / "result.json"

    if root.is_dir():
        unexpected = sorted(path.name for path in root.iterdir() if path.name not in ALLOWED_TOP_LEVEL)
        errors.extend(f"提交包顶层包含非官方条目：{name}" for name in unexpected)
        symlinks = sorted(
            path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink()
        )
        errors.extend(f"提交包不允许符号链接：{path}" for path in symlinks)

    code_files = [path for path in code_dir.rglob("*") if path.is_file()] if code_dir.is_dir() else []
    design_files = (
        [
            path
            for path in design_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in ALLOWED_DESIGN_SUFFIXES
        ]
        if design_dir.is_dir()
        else []
    )
    if not code_dir.is_dir():
        errors.append("缺少code目录")
    elif not code_files:
        errors.append("code目录为空")
    if not design_dir.is_dir():
        errors.append("缺少design目录")
    elif not design_files:
        errors.append("design目录缺少DOCX、PDF或Markdown方案文档")

    result_record_count = 0
    if not result_path.is_file():
        errors.append("缺少result/result.json")
    else:
        submission = validate_submission(result_path)
        result_record_count = submission.record_count
        errors.extend(f"result.json：{message}" for message in submission.errors)
    result_dir = root / "result"
    if result_dir.is_dir():
        unexpected_results = sorted(
            path.relative_to(result_dir).as_posix()
            for path in result_dir.rglob("*")
            if path.is_file() and path != result_path
        )
        errors.extend(f"result目录包含非官方文件：{path}" for path in unexpected_results)

    if root.is_dir():
        forbidden = [
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name.lower() in FORBIDDEN_NAMES
        ]
        errors.extend(f"提交包包含敏感配置文件：{path}" for path in forbidden)

    return DeliveryValidation(
        valid=not errors,
        errors=errors,
        code_file_count=len(code_files),
        design_file_count=len(design_files),
        result_record_count=result_record_count,
    )
