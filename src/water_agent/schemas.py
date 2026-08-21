from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from water_agent.labels import WaterLabel


class ClassScore(BaseModel):
    label: WaterLabel
    probability: float = Field(ge=0.0, le=1.0)


class ClassificationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: WaterLabel
    confidence: float = Field(ge=0.0, le=1.0)
    top_k: list[ClassScore] = Field(min_length=1)
    requires_review: bool
    review_reason: str | None = None
    model_version: str
    latency_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def predicted_label_must_be_top_one(self) -> ClassificationResult:
        if self.top_k[0].label != self.label:
            raise ValueError("top_k第一项必须与预测标签一致")
        return self


class ToolTrace(BaseModel):
    tool: str
    status: str
    latency_ms: float = Field(ge=0.0)


class TaskIntent(StrEnum):
    IMAGE_ASSESSMENT = "image_assessment"
    PATROL_REPORT = "patrol_report"
    RESULT_RELIABILITY = "result_reliability"
    LABEL_GUIDANCE = "label_guidance"


class PlannedTool(StrEnum):
    CLASSIFY_WATER_IMAGE = "classify_water_image"
    LOOKUP_LABEL_GUIDANCE = "lookup_label_guidance"
    ASSESS_REVIEW_NEED = "assess_review_need"
    COMPOSE_ASSESSMENT = "compose_assessment"
    COMPOSE_REPORT = "compose_report"


class TaskPlan(BaseModel):
    intent: TaskIntent
    tools: list[PlannedTool] = Field(min_length=1, max_length=4)
    reason: str = Field(min_length=1, max_length=160)
    planner_mode: str = Field(pattern=r"^(qwen|deterministic)$")

    @model_validator(mode="after")
    def tools_must_match_intent(self) -> TaskPlan:
        expected = {
            TaskIntent.IMAGE_ASSESSMENT: [
                PlannedTool.CLASSIFY_WATER_IMAGE,
                PlannedTool.COMPOSE_ASSESSMENT,
            ],
            TaskIntent.PATROL_REPORT: [
                PlannedTool.CLASSIFY_WATER_IMAGE,
                PlannedTool.LOOKUP_LABEL_GUIDANCE,
                PlannedTool.COMPOSE_REPORT,
            ],
            TaskIntent.RESULT_RELIABILITY: [
                PlannedTool.CLASSIFY_WATER_IMAGE,
                PlannedTool.ASSESS_REVIEW_NEED,
                PlannedTool.COMPOSE_ASSESSMENT,
            ],
            TaskIntent.LABEL_GUIDANCE: [PlannedTool.LOOKUP_LABEL_GUIDANCE],
        }
        if self.tools != expected[self.intent]:
            raise ValueError("工具序列不符合受控任务协议")
        return self


class LabelGuidance(BaseModel):
    label: WaterLabel
    definition: str
    common_confusions: list[str] = Field(min_length=1)
    review_checks: list[str] = Field(min_length=1)
    knowledge_version: str


class AnalysisResponse(BaseModel):
    answer: str
    result: ClassificationResult | None = None
    trace: list[ToolTrace]
    plan: TaskPlan | None = None
    guidance: LabelGuidance | None = None


class SubmissionRecord(BaseModel):
    filename: str = Field(min_length=1)
    width: str = Field(pattern=r"^\d+$")
    height: str = Field(pattern=r"^\d+$")
    label: WaterLabel

    @field_validator("width", "height")
    @classmethod
    def dimension_must_be_positive(cls, value: str) -> str:
        if int(value) <= 0:
            raise ValueError("尺寸必须大于0")
        return value


class SubmissionValidation(BaseModel):
    valid: bool
    record_count: int = Field(ge=0)
    errors: list[str]
