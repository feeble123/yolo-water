from __future__ import annotations

import sqlite3

from water_agent.agent import WaterAnalysisAgent
from water_agent.agent.qwen_provider import QwenChatProvider
from water_agent.audit import SQLiteAuditLogger
from water_agent.config import Settings
from water_agent.vision import FakeClassifier, UltralyticsClassifier, UltralyticsEnsembleClassifier


def build_agent(settings: Settings) -> WaterAnalysisAgent:
    if settings.use_fake_model:
        classifier = FakeClassifier(settings.review_threshold)
    elif settings.ensemble_weights:
        classifier = UltralyticsEnsembleClassifier(
            settings.ensemble_weights,
            settings.review_threshold,
            settings.ensemble_temperature,
            set(settings.force_review_labels),
            settings.vision_image_size,
            settings.vision_device,
        )
    else:
        classifier = UltralyticsClassifier(
            settings.model_weights,
            settings.review_threshold,
            set(settings.force_review_labels),
            settings.vision_image_size,
            settings.vision_device,
        )
    provider = QwenChatProvider(settings) if settings.use_qwen else None
    audit_sink = None
    if settings.enable_audit_log:
        try:
            audit_sink = SQLiteAuditLogger(settings.audit_db_path)
        except (OSError, sqlite3.Error):
            # 审计目录不可写时保留核心识别能力，实际状态通过health接口暴露。
            audit_sink = None
    return WaterAnalysisAgent(
        classifier=classifier,
        explanation_provider=provider,
        audit_sink=audit_sink,
    )
