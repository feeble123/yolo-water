from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def run_training(
    *,
    data_dir: Path,
    model: str,
    output_dir: Path,
    run_name: str,
    epochs: int,
    image_size: int,
    batch: int,
    device: str,
    workers: int,
    seed: int,
    class_weight_power: float = 0.0,
    class_weight_cap: float = 8.0,
    label_smoothing: float = 0.0,
) -> dict[str, Any]:
    if not (data_dir / "train").is_dir() or not (data_dir / "val").is_dir():
        raise FileNotFoundError("data目录必须包含train与val子目录")
    if epochs <= 0 or image_size <= 0 or batch <= 0 or workers < 0:
        raise ValueError("训练参数必须为正数，workers可为0")
    if not 0.0 <= class_weight_power <= 1.0:
        raise ValueError("class_weight_power必须在0到1之间")
    if class_weight_cap < 1.0:
        raise ValueError("class_weight_cap不能小于1")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing必须在[0, 1)之间")

    config_dir = output_dir.parent / "ultralytics_config"
    matplotlib_dir = output_dir.parent / "matplotlib_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir.resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir.resolve()))
    if os.name == "nt" and "WINDIR" not in os.environ:
        os.environ["WINDIR"] = os.environ.get("SystemRoot", r"C:\Windows")

    from ultralytics import YOLO

    trainer = YOLO(model)
    trainer_class = None
    if class_weight_power > 0.0:
        from water_agent.vision.long_tail import make_weighted_classification_trainer

        trainer_class = make_weighted_classification_trainer(
            power=class_weight_power,
            cap=class_weight_cap,
            label_smoothing=label_smoothing,
        )
    metrics = trainer.train(
        trainer=trainer_class,
        data=str(data_dir.resolve()),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        device=device,
        workers=workers,
        seed=seed,
        deterministic=True,
        amp=True,
        cache=False,
        project=str(output_dir.resolve()),
        name=run_name,
        exist_ok=False,
        plots=True,
        verbose=True,
    )
    save_dir = Path(metrics.save_dir)
    return {
        "save_dir": str(save_dir),
        "best_weights": str(save_dir / "weights" / "best.pt"),
        "last_weights": str(save_dir / "weights" / "last.pt"),
        "epochs": epochs,
        "image_size": image_size,
        "batch": batch,
        "device": device,
        "class_weight_power": class_weight_power,
        "class_weight_cap": class_weight_cap,
        "label_smoothing": label_smoothing,
        "note": "power为0时使用标准交叉熵；大于0时使用有上限的逆频率幂次类别权重。",
    }
