from __future__ import annotations

from typing import Final

from PIL import Image
from torchvision import transforms

IMAGENET_MEAN: Final = (0.485, 0.456, 0.406)
IMAGENET_STD: Final = (0.229, 0.224, 0.225)


class SquareLetterbox:
    """Resize an RGB PIL image without cropping, then pad it to a square canvas."""

    def __init__(self, size: int, fill: tuple[int, int, int] = (114, 114, 114)) -> None:
        if size <= 0:
            raise ValueError("size必须为正数")
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        scale = min(self.size / width, self.size / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = rgb.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(resized, ((self.size - resized_width) // 2, (self.size - resized_height) // 2))
        return canvas


def build_full_frame_transform(*, size: int, train: bool, horizontal_flip: float) -> transforms.Compose:
    """Create a conservative classification transform that never crops source pixels."""
    if not 0.0 <= horizontal_flip <= 1.0:
        raise ValueError("horizontal_flip必须在0到1之间")
    steps: list[object] = [SquareLetterbox(size)]
    if train and horizontal_flip:
        steps.append(transforms.RandomHorizontalFlip(p=horizontal_flip))
    steps.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(steps)
