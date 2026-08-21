from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from water_agent.agent import WaterAnalysisAgent
from water_agent.agent.qwen_provider import QwenProviderError
from water_agent.api.main import create_app
from water_agent.config import Settings
from water_agent.vision import FakeClassifier


def _jpeg_bytes(size: tuple[int, int] = (32, 32)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, "blue").save(buffer, format="JPEG")
    return buffer.getvalue()


def _client(**overrides: object) -> TestClient:
    settings = Settings(
        _env_file=None,
        use_fake_model=True,
        use_qwen=False,
        enable_audit_log=False,
        **overrides,
    )
    return TestClient(create_app(settings=settings))


def test_health_and_analyze_contract() -> None:
    client = _client()

    health = client.get("/health")
    analyze = client.post(
        "/api/v1/analyze",
        files={"image": ("sample.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"prompt": "请分析"},
    )

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "vision_mode": "fake",
        "qwen_enabled": False,
        "planner_mode": "deterministic",
        "knowledge_version": "water-label-guidance-v1",
        "audit_enabled": False,
    }
    assert analyze.status_code == 200
    body = analyze.json()
    assert body["result"]["label"] in {"乱采", "乱建", "乱堆", "乱占", "有漂浮物", "正常"}
    assert body["trace"][0]["tool"] == "plan_task"
    assert body["trace"][1]["tool"] == "classify_water_image"
    assert body["plan"]["planner_mode"] == "deterministic"


def test_label_guidance_endpoint_is_curated_and_path_free() -> None:
    response = _client().get("/api/v1/guidance/%E4%B9%B1%E5%A0%86")

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "乱堆"
    assert body["knowledge_version"] == "water-label-guidance-v1"
    assert "路径" not in str(body)


def test_consult_endpoint_uses_agent_plan_without_an_image() -> None:
    response = _client().post("/api/v1/consult", data={"prompt": "乱堆是什么意思？"})

    assert response.status_code == 200
    body = response.json()
    assert body["result"] is None
    assert body["plan"]["intent"] == "label_guidance"
    assert body["guidance"]["label"] == "乱堆"


def test_upload_rejects_bad_extension_and_corrupt_image() -> None:
    client = _client()

    bad_extension = client.post(
        "/api/v1/classify", files={"image": ("bad.txt", b"not-image", "text/plain")}
    )
    corrupt_image = client.post(
        "/api/v1/classify", files={"image": ("bad.jpg", b"not-image", "image/jpeg")}
    )

    assert bad_extension.status_code == 415
    assert corrupt_image.status_code == 415


def test_upload_rejects_excessive_pixel_count() -> None:
    client = _client(max_image_pixels=100)

    response = client.post(
        "/api/v1/classify",
        files={"image": ("large.jpg", _jpeg_bytes((32, 32)), "image/jpeg")},
    )

    assert response.status_code == 413
    assert "像素" in response.json()["detail"]


def test_analyze_degrades_to_vision_result_when_qwen_fails() -> None:
    class UnavailableProvider:
        def explain(self, _prompt: str, _result: object) -> str:
            raise QwenProviderError("Qwen解释服务暂时不可用", "APITimeoutError")

    settings = Settings(
        _env_file=None,
        use_fake_model=True,
        use_qwen=True,
        enable_audit_log=False,
    )
    agent = WaterAnalysisAgent(FakeClassifier(), explanation_provider=UnavailableProvider())
    client = TestClient(create_app(settings=settings, agent=agent))

    response = client.post(
        "/api/v1/analyze",
        files={"image": ("sample.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"prompt": "请分析"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["trace"][2]["status"] == "fallback:APITimeoutError"
    assert "视觉工具" in body["answer"]
