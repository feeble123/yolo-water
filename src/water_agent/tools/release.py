from __future__ import annotations

import hashlib
import re
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

from pydantic import BaseModel, Field

from water_agent.tools.delivery import ALLOWED_DESIGN_SUFFIXES, validate_delivery
from water_agent.tools.submission import validate_submission

CODE_FILES = (
    ".env.example",
    ".env.production.example",
    "AI智能体设计方案.docx",
    "README.md",
    "DEPLOYMENT.md",
    "MODEL_CARD.md",
    "LOCAL_BATCH_INFERENCE_GUIDE.md",
    "PHASE5_DELIVERY_REPORT.md",
    "THIRD_PARTY_NOTICES.md",
    "DEMO_SCRIPT.md",
    "RELEASE_CHECKLIST.md",
    "architecture.md",
    "acceptance_report.md",
    "pyproject.toml",
)
CODE_DIRECTORIES = ("src", "configs", "docs", "scripts", "tests")
MODEL_FILES = {
    "fold0.pt": "artifacts/training/ablation_fold0_weighted_p05_c8_e15/weights/best.pt",
    "fold1.pt": "artifacts/training/fold1_weighted_p05_c8_e15/weights/best.pt",
    "fold2.pt": "artifacts/training/fold2_weighted_p05_c8_e15/weights/best.pt",
    "fold3.pt": "artifacts/training/fold3_weighted_p05_c8_e15/weights/best.pt",
}
IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git", ".venv"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
DESIGN_PLACEHOLDERS = ("提交前填写报名队伍名", "【提交前填写", "参赛队伍：________")
SECRET_ASSIGNMENT = re.compile(
    r"(?im)^[ \t]*(?:WATER_AGENT_QWEN_API_KEY|API_KEY|TOKEN|AUTHORIZATION)"
    r"[ \t]*[:=][ \t]*(?!$|<|\{)([^ \t\r\n#]+)"
)
PYTHON_SECRET_LITERAL = re.compile(
    r'''(?im)\b(?:WATER_AGENT_QWEN_API_KEY|API_KEY|TOKEN|AUTHORIZATION)\b'''
    r'''[ \t]*=[ \t]*(?:[rubf]{0,2})?["'](?![<{])[^"'\r\n]+["']'''
)
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".txt",
    ".ps1",
    ".example",
}


class ReleaseBuildResult(BaseModel):
    archive: str
    archive_sha256: str
    result_record_count: int = Field(gt=0)
    code_file_count: int = Field(gt=0)
    design_file_count: int = Field(gt=0)


def _copy_directory(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise ValueError(f"白名单源码目录不能是符号链接：{source}")

    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name in IGNORED_PARTS or Path(name).suffix.lower() in IGNORED_SUFFIXES
        }

    shutil.copytree(source, destination, ignore=ignore)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(code_dir: Path) -> None:
    rows = []
    for path in sorted(item for item in code_dir.rglob("*") if item.is_file()):
        if path.name == "MANIFEST.sha256":
            continue
        rows.append(f"{_sha256(path)}  {path.relative_to(code_dir).as_posix()}")
    (code_dir / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _design_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        with zipfile.ZipFile(path) as archive:
            return archive.read("word/document.xml").decode("utf-8", errors="replace")
    if path.suffix.lower() in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _assert_no_placeholders(design_sources: list[Path]) -> None:
    for path in design_sources:
        text = _design_text(path)
        if any(marker in text for marker in DESIGN_PLACEHOLDERS):
            raise ValueError(f"方案文档仍含身份占位符：{path.name}")


def _assert_no_secret_assignments(stage: Path) -> None:
    for path in stage.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        matcher = PYTHON_SECRET_LITERAL if path.suffix.lower() == ".py" else SECRET_ASSIGNMENT
        if matcher.search(text):
            relative = path.relative_to(stage).as_posix()
            raise ValueError(f"提交包疑似包含已赋值密钥或令牌：{relative}")


def build_release(
    project_root: Path,
    result_path: Path,
    output_path: Path,
    *,
    confirm_real_result: bool,
) -> ReleaseBuildResult:
    project_root = project_root.resolve()
    result_path = result_path.resolve()
    output_path = output_path.resolve()

    if not confirm_real_result:
        raise ValueError(
            "拒绝构建：请仅在参赛者已对官方测试图片完成真实推理后，"
            "显式传入--confirm-real-result"
        )
    if "".join(output_path.suffixes[-2:]).lower() != ".tar.gz":
        raise ValueError("输出文件必须使用.tar.gz扩展名")
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有归档：{output_path}")

    submission = validate_submission(result_path)
    if not submission.valid or submission.record_count == 0:
        details = "；".join(submission.errors) or "结果数组为空"
        raise ValueError(f"result.json校验失败：{details}")

    missing = [name for name in CODE_FILES if not (project_root / name).is_file()]
    missing.extend(name for name in CODE_DIRECTORIES if not (project_root / name).is_dir())
    missing.extend(source for source in MODEL_FILES.values() if not (project_root / source).is_file())
    if missing:
        raise FileNotFoundError(f"交付白名单文件缺失：{missing}")

    design_sources = sorted(
        path
        for path in (project_root / "design").iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_DESIGN_SUFFIXES
    )
    if not design_sources:
        raise FileNotFoundError("design目录中没有DOCX、PDF或Markdown方案文档")
    _assert_no_placeholders(design_sources)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="water-agent-release-") as temp_dir:
        stage = Path(temp_dir) / "submission"
        code_dir = stage / "code"
        design_dir = stage / "design"
        result_dir = stage / "result"
        code_dir.mkdir(parents=True)
        design_dir.mkdir()
        result_dir.mkdir()

        for relative in CODE_FILES:
            shutil.copy2(project_root / relative, code_dir / relative)
        for relative in CODE_DIRECTORIES:
            _copy_directory(project_root / relative, code_dir / relative)

        models_dir = code_dir / "models"
        models_dir.mkdir()
        for target_name, source in MODEL_FILES.items():
            shutil.copy2(project_root / source, models_dir / target_name)
        for source in design_sources:
            shutil.copy2(source, design_dir / source.name)
        shutil.copy2(result_path, result_dir / "result.json")
        _assert_no_secret_assignments(stage)
        _write_manifest(code_dir)

        validation = validate_delivery(stage)
        if not validation.valid:
            raise ValueError(f"交付目录校验失败：{'；'.join(validation.errors)}")

        try:
            with tarfile.open(output_path, mode="w:gz") as archive:
                for directory in ("code", "design", "result"):
                    archive.add(stage / directory, arcname=directory, recursive=True)
        except Exception:
            output_path.unlink(missing_ok=True)
            raise

    return ReleaseBuildResult(
        archive=str(output_path),
        archive_sha256=_sha256(output_path),
        result_record_count=submission.record_count,
        code_file_count=validation.code_file_count,
        design_file_count=validation.design_file_count,
    )
