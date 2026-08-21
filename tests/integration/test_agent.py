from pathlib import Path

from PIL import Image

from water_agent.agent import WaterAnalysisAgent
from water_agent.agent.qwen_provider import QwenProviderError
from water_agent.vision import FakeClassifier


def test_agent_calls_classifier_and_returns_trace(tmp_path: Path) -> None:
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 32), color="blue").save(path)
    agent = WaterAnalysisAgent(FakeClassifier(review_threshold=0.65))
    response = agent.analyze(path, "请分析")
    assert response.result.model_version == "fake-architecture-test-only"
    assert response.trace[0].tool == "plan_task"
    assert response.trace[0].status == "deterministic"
    assert response.trace[1].tool == "classify_water_image"
    assert response.plan is not None
    assert response.plan.planner_mode == "deterministic"
    assert "置信度" in response.answer


def test_agent_keeps_vision_result_when_qwen_is_unavailable(tmp_path: Path) -> None:
    class UnavailableProvider:
        def explain(self, _prompt: str, _result: object) -> str:
            raise QwenProviderError("Qwen解释服务暂时不可用", "APITimeoutError")

    path = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 32), color="blue").save(path)
    agent = WaterAnalysisAgent(
        FakeClassifier(review_threshold=0.65), explanation_provider=UnavailableProvider()
    )

    response = agent.analyze(path, "请分析")

    assert response.result.model_version == "fake-architecture-test-only"
    assert response.trace[2].tool == "compose_assessment"
    assert response.trace[2].status == "fallback:APITimeoutError"
    assert "视觉工具" in response.answer


def test_agent_can_consult_label_guidance_without_an_image() -> None:
    agent = WaterAnalysisAgent(FakeClassifier())

    response = agent.consult("乱堆是什么意思？")

    assert response.result is None
    assert response.plan is not None
    assert response.plan.intent.value == "label_guidance"
    assert response.guidance is not None
    assert response.guidance.label.value == "乱堆"
    assert [item.tool for item in response.trace] == ["plan_task", "lookup_label_guidance"]


def test_agent_resolves_common_label_aliases_without_an_image() -> None:
    agent = WaterAnalysisAgent(FakeClassifier())

    response = agent.consult("漂浮物是什么意思？")

    assert response.guidance is not None
    assert response.guidance.label.value == "有漂浮物"
