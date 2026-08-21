from __future__ import annotations

import json
from pathlib import Path

from water_agent.tools.delivery import validate_delivery


def _write_valid_result(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            [
                {
                    "filename": "00001.jpg",
                    "width": "1920",
                    "height": "1080",
                    "label": "正常",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_valid_delivery_structure(tmp_path: Path) -> None:
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "main.py").write_text("print('ok')", encoding="utf-8")
    (tmp_path / "design").mkdir()
    (tmp_path / "design" / "方案.md").write_text("# 方案", encoding="utf-8")
    _write_valid_result(tmp_path / "result" / "result.json")

    result = validate_delivery(tmp_path)

    assert result.valid
    assert result.result_record_count == 1


def test_delivery_rejects_missing_sections_and_secret_files(tmp_path: Path) -> None:
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "api key.txt").write_text("placeholder", encoding="utf-8")

    result = validate_delivery(tmp_path)

    assert not result.valid
    assert any("design" in error for error in result.errors)
    assert any("result/result.json" in error for error in result.errors)
    assert any("敏感配置" in error for error in result.errors)


def test_delivery_rejects_unexpected_top_level_and_result_files(tmp_path: Path) -> None:
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "main.py").write_text("pass", encoding="utf-8")
    (tmp_path / "design").mkdir()
    (tmp_path / "design" / "方案.md").write_text("# 方案", encoding="utf-8")
    _write_valid_result(tmp_path / "result" / "result.json")
    (tmp_path / "result" / "result.audit.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("unexpected", encoding="utf-8")

    result = validate_delivery(tmp_path)

    assert not result.valid
    assert any("顶层包含非官方条目" in error for error in result.errors)
    assert any("result目录包含非官方文件" in error for error in result.errors)
