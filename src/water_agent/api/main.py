from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from water_agent.agent import WaterAnalysisAgent
from water_agent.agent.knowledge import KNOWLEDGE_VERSION, lookup_label_guidance
from water_agent.config import Settings, get_settings
from water_agent.labels import WaterLabel
from water_agent.runtime import build_agent
from water_agent.schemas import AnalysisResponse, ClassificationResult, LabelGuidance

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _validated_temp_image(upload: UploadFile, data: bytes, settings: Settings) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="仅支持JPG、PNG或WebP图片")
    if not data or len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="图片为空或超过大小限制")
    with tempfile.NamedTemporaryFile(prefix="water-agent-", suffix=suffix, delete=False) as handle:
        path = Path(handle.name)
        handle.write(data)
    try:
        with Image.open(path) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > settings.max_image_pixels:
                raise HTTPException(status_code=413, detail="图片像素尺寸不合法或超过限制")
            image.verify()
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except (UnidentifiedImageError, OSError):
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=415, detail="文件不是可解码图片") from None
    return path


def create_app(settings: Settings | None = None, agent: WaterAnalysisAgent | None = None) -> FastAPI:
    settings = settings or get_settings()
    agent = agent or build_agent(settings)
    app = FastAPI(title="水域综合异常识别智能体", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, object]:
        vision_mode = (
            "fake"
            if settings.use_fake_model
            else "ensemble"
            if settings.ensemble_weights
            else "ultralytics"
        )
        return {
            "status": "ok",
            "vision_mode": vision_mode,
            "qwen_enabled": settings.use_qwen,
            "planner_mode": "qwen" if settings.use_qwen else "deterministic",
            "knowledge_version": KNOWLEDGE_VERSION,
            "audit_enabled": agent.audit_sink is not None,
        }

    @app.get("/api/v1/guidance/{label}", response_model=LabelGuidance)
    def label_guidance(label: WaterLabel) -> LabelGuidance:
        """Return curated category guidance without accepting any local file path."""
        return lookup_label_guidance(label)

    @app.post("/api/v1/consult", response_model=AnalysisResponse)
    def consult(prompt: Annotated[str, Form()]) -> AnalysisResponse:
        """Answer a label-guidance question without accepting an image or local path."""
        return agent.consult(prompt.strip() or "请说明有漂浮物的含义")

    @app.post("/api/v1/analyze", response_model=AnalysisResponse)
    async def analyze(
        image: Annotated[UploadFile, File()],
        prompt: Annotated[str, Form()] = "请分析这张图片中的水域异常",
    ) -> AnalysisResponse:
        data = await image.read(settings.max_upload_bytes + 1)
        path = _validated_temp_image(image, data, settings)
        try:
            return agent.analyze(path, prompt.strip() or "请分析这张图片中的水域异常")
        finally:
            path.unlink(missing_ok=True)

    @app.post("/api/v1/classify", response_model=ClassificationResult)
    async def classify(image: Annotated[UploadFile, File()]) -> ClassificationResult:
        data = await image.read(settings.max_upload_bytes + 1)
        path = _validated_temp_image(image, data, settings)
        try:
            return agent.classifier.classify(path)
        finally:
            path.unlink(missing_ok=True)

    return app


app = create_app()
