from __future__ import annotations

import pytest
from PIL import Image

from water_agent.vision.full_frame import SquareLetterbox, build_full_frame_transform


def test_square_letterbox_preserves_aspect_ratio_and_pads() -> None:
    image = Image.new("RGB", (8, 4), (255, 0, 0))

    result = SquareLetterbox(10)(image)

    assert result.size == (10, 10)
    assert result.getpixel((5, 0)) == (114, 114, 114)
    assert result.getpixel((5, 5)) == (255, 0, 0)


def test_full_frame_transform_returns_normalized_square_tensor() -> None:
    transform = build_full_frame_transform(size=12, train=False, horizontal_flip=0.5)

    result = transform(Image.new("RGB", (3, 9), (128, 128, 128)))

    assert tuple(result.shape) == (3, 12, 12)


def test_full_frame_transform_rejects_invalid_flip_probability() -> None:
    with pytest.raises(ValueError, match="horizontal_flip"):
        build_full_frame_transform(size=12, train=True, horizontal_flip=1.1)
