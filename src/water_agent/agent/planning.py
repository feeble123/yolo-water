from __future__ import annotations

from typing import Protocol

from water_agent.agent.qwen_provider import QwenProviderError
from water_agent.schemas import PlannedTool, TaskIntent, TaskPlan


class TaskPlanner(Protocol):
    def plan_task(self, prompt: str, image_available: bool) -> TaskPlan: ...


def deterministic_plan(prompt: str, image_available: bool) -> TaskPlan:
    normalized = prompt.strip()
    if not image_available:
        return TaskPlan(
            intent=TaskIntent.LABEL_GUIDANCE,
            tools=[PlannedTool.LOOKUP_LABEL_GUIDANCE],
            reason="未提供图片，改为回答赛题类别说明。",
            planner_mode="deterministic",
        )
    if any(keyword in normalized for keyword in ("报告", "建议", "处置", "巡查")):
        return TaskPlan(
            intent=TaskIntent.PATROL_REPORT,
            tools=[
                PlannedTool.CLASSIFY_WATER_IMAGE,
                PlannedTool.LOOKUP_LABEL_GUIDANCE,
                PlannedTool.COMPOSE_REPORT,
            ],
            reason="用户要求识别图片并生成巡查建议。",
            planner_mode="deterministic",
        )
    if any(keyword in normalized for keyword in ("可靠", "置信", "复核", "准确")):
        return TaskPlan(
            intent=TaskIntent.RESULT_RELIABILITY,
            tools=[
                PlannedTool.CLASSIFY_WATER_IMAGE,
                PlannedTool.ASSESS_REVIEW_NEED,
                PlannedTool.COMPOSE_ASSESSMENT,
            ],
            reason="用户关注识别结果的可靠性与复核需求。",
            planner_mode="deterministic",
        )
    return TaskPlan(
        intent=TaskIntent.IMAGE_ASSESSMENT,
        tools=[
            PlannedTool.CLASSIFY_WATER_IMAGE,
            PlannedTool.COMPOSE_ASSESSMENT,
        ],
        reason="用户要求分析图片中的水域异常。",
        planner_mode="deterministic",
    )


def plan_with_fallback(
    planner: TaskPlanner | None, prompt: str, image_available: bool
) -> tuple[TaskPlan, str]:
    if planner is None:
        return deterministic_plan(prompt, image_available), "deterministic"
    try:
        plan = planner.plan_task(prompt, image_available)
    except QwenProviderError as exc:
        return deterministic_plan(prompt, image_available), f"fallback:{exc.error_kind}"
    image_plan_is_valid = (plan.intent == TaskIntent.LABEL_GUIDANCE) == (not image_available)
    if plan.planner_mode != "qwen" or not image_plan_is_valid:
        return deterministic_plan(prompt, image_available), "fallback:invalid_plan"
    return plan, "success"
