from __future__ import annotations

import sqlite3
from pathlib import Path

from water_agent.audit import SQLiteAuditLogger
from water_agent.labels import WaterLabel
from water_agent.schemas import AnalysisResponse, ClassificationResult, ClassScore, ToolTrace


def test_sqlite_audit_excludes_prompt_and_image(tmp_path: Path) -> None:
    database = tmp_path / "audit.sqlite3"
    logger = SQLiteAuditLogger(database)
    response = AnalysisResponse(
        answer="安全解释",
        result=ClassificationResult(
            label=WaterLabel.NORMAL,
            confidence=0.8,
            top_k=[ClassScore(label=WaterLabel.NORMAL, probability=0.8)],
            requires_review=True,
            review_reason="验证证据不足",
            model_version="test",
            latency_ms=1,
        ),
        trace=[ToolTrace(tool="classify_water_image", status="success", latency_ms=1)],
    )

    request_id = logger.record(response)

    with sqlite3.connect(database) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(analysis_events)").fetchall()
        }
        row = connection.execute(
            "SELECT label, requires_review, model_version FROM analysis_events WHERE request_id=?",
            (request_id,),
        ).fetchone()
    assert "prompt" not in columns
    assert "image_path" not in columns
    assert row == ("正常", 1, "test")
