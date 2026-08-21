from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from water_agent.schemas import AnalysisResponse


class AuditSink(Protocol):
    def record(self, response: AnalysisResponse) -> str: ...


class SQLiteAuditLogger:
    """只记录结构化模型结果和工具轨迹，不保存图片、问题或密钥。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_events (
                    request_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    requires_review INTEGER NOT NULL,
                    review_reason TEXT,
                    model_version TEXT NOT NULL,
                    total_latency_ms REAL NOT NULL,
                    trace_json TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    def record(self, response: AnalysisResponse) -> str:
        request_id = uuid4().hex
        trace = [item.model_dump(mode="json") for item in response.trace]
        total_latency_ms = sum(item.latency_ms for item in response.trace)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO analysis_events (
                    request_id, created_at, label, confidence, requires_review,
                    review_reason, model_version, total_latency_ms, trace_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    datetime.now(UTC).isoformat(),
                    str(response.result.label),
                    response.result.confidence,
                    int(response.result.requires_review),
                    response.result.review_reason,
                    response.result.model_version,
                    total_latency_ms,
                    json.dumps(trace, ensure_ascii=False),
                ),
            )
        return request_id
