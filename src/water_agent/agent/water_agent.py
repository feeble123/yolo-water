from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from water_agent.agent.knowledge import lookup_label_guidance, resolve_label_from_text
from water_agent.agent.planning import TaskPlanner, plan_with_fallback
from water_agent.agent.qwen_provider import ExplanationProvider, QwenProviderError
from water_agent.audit import AuditSink
from water_agent.labels import LABEL_DESCRIPTIONS
from water_agent.schemas import (
    AnalysisResponse,
    ClassificationResult,
    LabelGuidance,
    PlannedTool,
    TaskIntent,
    ToolTrace,
)
from water_agent.vision import VisionClassifier


class WaterAnalysisAgent:
    def __init__(
        self,
        classifier: VisionClassifier,
        explanation_provider: ExplanationProvider | None = None,
        audit_sink: AuditSink | None = None,
        task_planner: TaskPlanner | None = None,
    ) -> None:
        self.classifier = classifier
        self.explanation_provider = explanation_provider
        self.audit_sink = audit_sink
        self.task_planner = task_planner
        if self.task_planner is None and hasattr(explanation_provider, "plan_task"):
            self.task_planner = explanation_provider  # type: ignore[assignment]

    @staticmethod
    def _fallback_answer(
        result: ClassificationResult,
        guidance: LabelGuidance | None = None,
        report: bool = False,
    ) -> str:
        label = result.label
        description = LABEL_DESCRIPTIONS[label]
        answer = f"识别结果为“{label}”，置信度{result.confidence:.1%}。{description}"
        if result.requires_review:
            answer += " 当前结果置信度较低，建议结合现场信息进行人工复核。"
        if report and guidance:
            answer += f" 建议：{guidance.review_checks[0]}"
        return answer

    def analyze(self, image_path: Path, prompt: str) -> AnalysisResponse:
        plan_started = time.perf_counter()
        plan, plan_status = plan_with_fallback(self.task_planner, prompt, image_available=True)
        trace = [
            ToolTrace(
                tool="plan_task",
                status=plan_status,
                latency_ms=(time.perf_counter() - plan_started) * 1000,
            )
        ]
        result = self.classifier.classify(image_path)
        trace.append(
            ToolTrace(
                tool="classify_water_image",
                status="success",
                latency_ms=result.latency_ms,
            )
        )
        guidance = None
        if PlannedTool.LOOKUP_LABEL_GUIDANCE in plan.tools:
            guidance_started = time.perf_counter()
            guidance = lookup_label_guidance(result.label)
            trace.append(
                ToolTrace(
                    tool="lookup_label_guidance",
                    status="success",
                    latency_ms=(time.perf_counter() - guidance_started) * 1000,
                )
            )
        if PlannedTool.ASSESS_REVIEW_NEED in plan.tools:
            review_started = time.perf_counter()
            trace.append(
                ToolTrace(
                    tool="assess_review_need",
                    status="success",
                    latency_ms=(time.perf_counter() - review_started) * 1000,
                )
            )
        report = plan.intent == TaskIntent.PATROL_REPORT
        if self.explanation_provider:
            qwen_started = time.perf_counter()
            try:
                if guidance or report:
                    answer = self.explanation_provider.explain(prompt, result, guidance, report)
                else:
                    # Keep third-party and older test explanation providers compatible.
                    answer = self.explanation_provider.explain(prompt, result)
                qwen_status = "success"
            except QwenProviderError as exc:
                answer = self._fallback_answer(result, guidance, report)
                answer += " 大模型解释服务暂时不可用，以上为视觉工具的结构化结果。"
                qwen_status = f"fallback:{exc.error_kind}"
            trace.append(
                ToolTrace(
                    tool="compose_report" if report else "compose_assessment",
                    status=qwen_status,
                    latency_ms=(time.perf_counter() - qwen_started) * 1000,
                )
            )
        else:
            compose_started = time.perf_counter()
            answer = self._fallback_answer(result, guidance, report)
            trace.append(
                ToolTrace(
                    tool="compose_report" if report else "compose_assessment",
                    status="success",
                    latency_ms=(time.perf_counter() - compose_started) * 1000,
                )
            )
        response = AnalysisResponse(
            answer=answer,
            result=result,
            trace=trace,
            plan=plan,
            guidance=guidance,
        )
        if self.audit_sink:
            audit_started = time.perf_counter()
            try:
                self.audit_sink.record(response)
                audit_status = "success"
            except (OSError, sqlite3.Error):
                # 审计存储故障不能覆盖已经完成的视觉结果，也不能把数据库细节返回给用户。
                audit_status = "failure"
            response.trace.append(
                ToolTrace(
                    tool="write_audit_log",
                    status=audit_status,
                    latency_ms=(time.perf_counter() - audit_started) * 1000,
                )
            )
        return response

    def consult(self, prompt: str) -> AnalysisResponse:
        """Answer a category guidance request without loading an image or model."""
        plan_started = time.perf_counter()
        plan, plan_status = plan_with_fallback(self.task_planner, prompt, image_available=False)
        trace = [
            ToolTrace(
                tool="plan_task",
                status=plan_status,
                latency_ms=(time.perf_counter() - plan_started) * 1000,
            )
        ]
        label = resolve_label_from_text(prompt)
        if label is None:
            trace.append(ToolTrace(tool="lookup_label_guidance", status="missing_label", latency_ms=0.0))
            return AnalysisResponse(
                answer="请在问题中明确六类中的一个类别，例如“乱堆是什么意思？”",
                result=None,
                trace=trace,
                plan=plan,
                guidance=None,
            )
        guidance_started = time.perf_counter()
        guidance = lookup_label_guidance(label)
        trace.append(
            ToolTrace(
                tool="lookup_label_guidance",
                status="success",
                latency_ms=(time.perf_counter() - guidance_started) * 1000,
            )
        )
        answer = (
            f"“{guidance.label}”指{guidance.definition} "
            f"易混淆点：{guidance.common_confusions[0]} "
            f"建议：{guidance.review_checks[0]}"
        )
        return AnalysisResponse(
            answer=answer,
            result=None,
            trace=trace,
            plan=plan,
            guidance=guidance,
        )
