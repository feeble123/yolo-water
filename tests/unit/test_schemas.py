import pytest
from pydantic import ValidationError

from water_agent.labels import WaterLabel
from water_agent.schemas import ClassificationResult, ClassScore


def test_top_one_must_match_label() -> None:
    with pytest.raises(ValidationError):
        ClassificationResult(
            label=WaterLabel.NORMAL,
            confidence=0.8,
            top_k=[ClassScore(label=WaterLabel.FLOATING_DEBRIS, probability=0.8)],
            requires_review=False,
            model_version="test",
            latency_ms=1.0,
        )
