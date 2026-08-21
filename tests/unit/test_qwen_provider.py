from __future__ import annotations

from types import SimpleNamespace

import pytest

from water_agent.agent.qwen_provider import QwenChatProvider, QwenProviderError
from water_agent.config import Settings
from water_agent.labels import WaterLabel
from water_agent.schemas import ClassificationResult, ClassScore


class StubClient:
    def __init__(self, content: str = "已根据视觉工具完成解释") -> None:
        self.content = content
        self.messages: list[tuple[str, str]] = []

    def invoke(self, messages: list[tuple[str, str]]) -> SimpleNamespace:
        self.messages = messages
        return SimpleNamespace(content=self.content)


def _result() -> ClassificationResult:
    return ClassificationResult(
        label=WaterLabel.FLOATING_DEBRIS,
        confidence=0.91,
        top_k=[ClassScore(label=WaterLabel.FLOATING_DEBRIS, probability=0.91)],
        requires_review=False,
        model_version="test",
        latency_ms=12.3,
    )


def test_provider_only_sends_structured_tool_result() -> None:
    client = StubClient()
    settings = Settings(_env_file=None, qwen_max_prompt_chars=4)
    provider = QwenChatProvider(settings, client=client)

    answer = provider.explain("123456", _result())

    assert answer == "已根据视觉工具完成解释"
    assert "1234" in client.messages[1][1]
    assert "123456" not in client.messages[1][1]
    assert "有漂浮物" in client.messages[1][1]
    assert "latency_ms" not in client.messages[1][1]


def test_provider_wraps_client_errors_without_exposing_details() -> None:
    class BrokenClient:
        def invoke(self, _messages: list[tuple[str, str]]) -> None:
            raise RuntimeError("secret upstream response")

    provider = QwenChatProvider(Settings(_env_file=None), client=BrokenClient())

    with pytest.raises(QwenProviderError, match="RuntimeError") as error:
        provider.explain("请分析", _result())
    assert "secret" not in str(error.value)
    assert error.value.error_kind == "RuntimeError"


def test_healthcheck_returns_no_model_content() -> None:
    provider = QwenChatProvider(Settings(_env_file=None), client=StubClient("OK"))

    health = provider.healthcheck()

    assert health["status"] == "ok"
    assert health["model"] == "qwen3.6-27b"
    assert "content" not in health


def test_provider_parses_constrained_task_plan() -> None:
    client = StubClient(
        '{"intent":"patrol_report","tools":["classify_water_image",'
        '"lookup_label_guidance","compose_report"],"reason":"需要巡查建议"}'
    )
    provider = QwenChatProvider(Settings(_env_file=None), client=client)

    plan = provider.plan_task("请生成巡查报告", image_available=True)

    assert plan.planner_mode == "qwen"
    assert plan.intent.value == "patrol_report"
    assert "只输出一个JSON对象" in client.messages[0][1]


def test_provider_rejects_invalid_task_plan() -> None:
    provider = QwenChatProvider(Settings(_env_file=None), client=StubClient("not-json"))

    with pytest.raises(QwenProviderError, match="InvalidPlan"):
        provider.plan_task("请分析", image_available=True)
