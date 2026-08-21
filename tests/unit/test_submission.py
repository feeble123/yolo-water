import json
from pathlib import Path

from water_agent.tools.submission import validate_submission


def test_valid_submission(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            [{"filename": "00001.jpg", "width": "1920", "height": "1080", "label": "正常"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = validate_submission(path)
    assert result.valid
    assert result.record_count == 1


def test_duplicate_filename_is_rejected(tmp_path: Path) -> None:
    record = {"filename": "00001.jpg", "width": "1920", "height": "1080", "label": "正常"}
    path = tmp_path / "result.json"
    path.write_text(json.dumps([record, record], ensure_ascii=False), encoding="utf-8")
    result = validate_submission(path)
    assert not result.valid
    assert any("重复" in error for error in result.errors)
