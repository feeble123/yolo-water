from water_agent.labels import WaterLabel
from water_agent.vision.classifier import _build_result


def _probabilities(top_label: WaterLabel, confidence: float) -> dict[WaterLabel, float]:
    remainder = (1.0 - confidence) / (len(WaterLabel) - 1)
    return {label: confidence if label == top_label else remainder for label in WaterLabel}


def test_normal_is_forced_to_review_despite_high_confidence() -> None:
    result = _build_result(
        _probabilities(WaterLabel.NORMAL, 0.8),
        threshold=0.65,
        model_version="test",
        latency_ms=1,
        force_review_labels={WaterLabel.NORMAL},
    )

    assert result.requires_review
    assert result.review_reason is not None
    assert "OOF召回率为0%" in result.review_reason


def test_reliable_class_uses_confidence_threshold() -> None:
    result = _build_result(
        _probabilities(WaterLabel.FLOATING_DEBRIS, 0.8),
        threshold=0.65,
        model_version="test",
        latency_ms=1,
        force_review_labels={WaterLabel.NORMAL},
    )

    assert not result.requires_review
    assert result.review_reason is None
