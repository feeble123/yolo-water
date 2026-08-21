from __future__ import annotations

from pathlib import Path

from water_agent.config import Settings


def test_settings_parse_ensemble_weights_from_environment(monkeypatch) -> None:
    monkeypatch.setenv(
        "WATER_AGENT_ENSEMBLE_WEIGHTS",
        '["artifacts/training/fold0/best.pt", "artifacts/training/fold1/best.pt"]',
    )

    settings = Settings(_env_file=None)

    assert settings.ensemble_weights == [
        Path("artifacts/training/fold0/best.pt"),
        Path("artifacts/training/fold1/best.pt"),
    ]


def test_settings_parse_real_vision_runtime_parameters(monkeypatch) -> None:
    monkeypatch.setenv("WATER_AGENT_VISION_IMAGE_SIZE", "320")
    monkeypatch.setenv("WATER_AGENT_VISION_DEVICE", "0")

    settings = Settings(_env_file=None)

    assert settings.vision_image_size == 320
    assert settings.vision_device == "0"
