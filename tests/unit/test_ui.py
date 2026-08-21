from __future__ import annotations

from pathlib import Path

from PIL import Image

from water_agent.agent import WaterAnalysisAgent
from water_agent.ui import analyze_for_display
from water_agent.vision import FakeClassifier


def test_display_exposes_prediction_and_trace(tmp_path: Path) -> None:
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 32), "blue").save(image)
    agent = WaterAnalysisAgent(FakeClassifier())

    answer, prediction, review, plan, guidance, top_k, trace = analyze_for_display(
        agent, str(image), "请分析"
    )

    assert answer
    assert "（" in prediction
    assert review in {"是", "否"} or review.startswith("是：")
    assert "候选类别" in top_k
    assert top_k.count("|") >= 12
    assert "任务类型" in plan
    assert "本任务不需要" in guidance
    assert "plan_task" in trace
    assert "classify_water_image" in trace


def test_display_supports_label_consultation_without_image() -> None:
    agent = WaterAnalysisAgent(FakeClassifier())

    answer, prediction, review, plan, guidance, top_k, trace = analyze_for_display(
        agent, None, "乱堆是什么意思？"
    )

    assert "乱堆" in answer
    assert prediction == "无需图片"
    assert review == "不适用"
    assert "label_guidance" in plan
    assert "类别：** 乱堆" in guidance
    assert "未调用YOLO" in top_k
    assert "lookup_label_guidance" in trace
