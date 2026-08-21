from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from water_agent.tools.release import CODE_DIRECTORIES, CODE_FILES, MODEL_FILES, build_release


def _project_fixture(root: Path) -> Path:
    for relative in CODE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture: {relative}", encoding="utf-8")
    for relative in CODE_DIRECTORIES:
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "keep.txt").write_text("keep", encoding="utf-8")
        cache = directory / "__pycache__"
        cache.mkdir()
        (cache / "drop.pyc").write_bytes(b"cache")
    for relative in MODEL_FILES.values():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")
    design = root / "design"
    design.mkdir()
    (design / "方案.md").write_text("# 方案", encoding="utf-8")
    result = root / "result.json"
    result.write_text(
        json.dumps(
            [{"filename": "00001.jpg", "width": "10", "height": "10", "label": "正常"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return result


def test_build_release_uses_only_three_top_level_directories(tmp_path: Path) -> None:
    result = _project_fixture(tmp_path)
    archive = tmp_path / "submission.tar.gz"

    built = build_release(
        tmp_path,
        result,
        archive,
        confirm_real_result=True,
    )

    assert built.result_record_count == 1
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert {name.split("/", 1)[0] for name in names} == {"code", "design", "result"}
    assert "code/MANIFEST.sha256" in names
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)


def test_build_release_requires_explicit_real_result_confirmation(tmp_path: Path) -> None:
    result = _project_fixture(tmp_path)
    archive = tmp_path / "submission.tar.gz"

    with pytest.raises(ValueError, match="confirm-real-result"):
        build_release(
            tmp_path,
            result,
            archive,
            confirm_real_result=False,
        )

    assert not archive.exists()


def test_build_release_rejects_design_identity_placeholder(tmp_path: Path) -> None:
    result = _project_fixture(tmp_path)
    (tmp_path / "design" / "方案.md").write_text(
        "参赛队伍：【提交前填写报名队伍名】", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="身份占位符"):
        build_release(
            tmp_path,
            result,
            tmp_path / "submission.tar.gz",
            confirm_real_result=True,
        )


def test_build_release_rejects_assigned_api_key(tmp_path: Path) -> None:
    result = _project_fixture(tmp_path)
    (tmp_path / ".env.example").write_text(
        "WATER_AGENT_QWEN_API_KEY=definitely-not-a-real-secret", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="密钥或令牌"):
        build_release(
            tmp_path,
            result,
            tmp_path / "submission.tar.gz",
            confirm_real_result=True,
        )


def test_build_release_allows_blank_api_key_example(tmp_path: Path) -> None:
    result = _project_fixture(tmp_path)
    (tmp_path / ".env.example").write_text(
        "WATER_AGENT_QWEN_API_KEY=\nWATER_AGENT_QWEN_API_KEY_FILE=\n",
        encoding="utf-8",
    )

    built = build_release(
        tmp_path,
        result,
        tmp_path / "submission.tar.gz",
        confirm_real_result=True,
    )

    assert built.result_record_count == 1


def test_build_release_allows_runtime_python_secret_lookup(tmp_path: Path) -> None:
    result = _project_fixture(tmp_path)
    (tmp_path / "src" / "runtime_secret.py").write_text(
        "api_key=secret.get_secret_value()\n", encoding="utf-8"
    )

    built = build_release(
        tmp_path,
        result,
        tmp_path / "submission.tar.gz",
        confirm_real_result=True,
    )

    assert built.result_record_count == 1


def test_build_release_rejects_python_literal_api_key(tmp_path: Path) -> None:
    result = _project_fixture(tmp_path)
    hardcoded_assignment = "api" + '_key="definitely-not-a-real-secret"\n'
    (tmp_path / "src" / "hardcoded_secret.py").write_text(
        hardcoded_assignment, encoding="utf-8"
    )

    with pytest.raises(ValueError, match="密钥或令牌"):
        build_release(
            tmp_path,
            result,
            tmp_path / "submission.tar.gz",
            confirm_real_result=True,
        )
