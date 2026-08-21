from __future__ import annotations

from water_agent.agent.planning import deterministic_plan, plan_with_fallback
from water_agent.agent.qwen_provider import QwenProviderError
from water_agent.schemas import PlannedTool, TaskIntent, TaskPlan


def test_deterministic_plan_selects_report_tools() -> None:
    plan = deterministic_plan("请生成巡查处置建议", image_available=True)

    assert plan.intent == TaskIntent.PATROL_REPORT
    assert plan.tools == [
        PlannedTool.CLASSIFY_WATER_IMAGE,
        PlannedTool.LOOKUP_LABEL_GUIDANCE,
        PlannedTool.COMPOSE_REPORT,
    ]


def test_no_image_uses_only_label_guidance() -> None:
    plan = deterministic_plan("乱堆是什么意思？", image_available=False)

    assert plan.intent == TaskIntent.LABEL_GUIDANCE
    assert plan.tools == [PlannedTool.LOOKUP_LABEL_GUIDANCE]


def test_invalid_qwen_image_plan_falls_back() -> None:
    class InvalidPlanner:
        def plan_task(self, _prompt: str, _image_available: bool) -> TaskPlan:
            return TaskPlan(
                intent=TaskIntent.LABEL_GUIDANCE,
                tools=[PlannedTool.LOOKUP_LABEL_GUIDANCE],
                reason="错误地忽略了已上传图片。",
                planner_mode="qwen",
            )

    plan, status = plan_with_fallback(InvalidPlanner(), "请识别", image_available=True)

    assert status == "fallback:invalid_plan"
    assert plan.intent == TaskIntent.IMAGE_ASSESSMENT


def test_qwen_error_falls_back_to_deterministic_plan() -> None:
    class BrokenPlanner:
        def plan_task(self, _prompt: str, _image_available: bool) -> TaskPlan:
            raise QwenProviderError("服务不可用", "APITimeoutError")

    plan, status = plan_with_fallback(BrokenPlanner(), "请识别", image_available=True)

    assert status == "fallback:APITimeoutError"
    assert plan.planner_mode == "deterministic"
