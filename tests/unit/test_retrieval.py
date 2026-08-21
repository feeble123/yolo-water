from __future__ import annotations

import numpy as np
import pytest

from water_agent.vision.retrieval import fuse_probabilities, knn_class_probabilities


def test_knn_retrieval_returns_the_nearest_class() -> None:
    bank = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    query = np.asarray([[0.9, 0.1], [0.1, 0.9]])

    scores = knn_class_probabilities(query, bank, ["甲", "乙"], ["甲", "乙"], k=1)

    assert scores.argmax(axis=1).tolist() == [0, 1]
    assert scores.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_fusion_uses_requested_weight() -> None:
    classifier = np.asarray([[0.8, 0.2]])
    retrieval = np.asarray([[0.2, 0.8]])

    assert fuse_probabilities(classifier, retrieval, alpha=0.25).tolist()[0] == pytest.approx(
        [0.65, 0.35]
    )
