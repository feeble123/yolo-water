from PIL import Image

from water_agent.vision.data_audit import difference_hash


def test_difference_hash_is_stable_for_identical_images() -> None:
    first = Image.new("RGB", (20, 20), "white")
    second = Image.new("RGB", (20, 20), "white")
    assert difference_hash(first) == difference_hash(second)

