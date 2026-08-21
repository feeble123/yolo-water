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
    image_transform: str = "default",
    loss_strategy: str = "auto",
    logit_adjustment_tau: float = 0.0,
    ldam_max_margin: float = 0.5,
    ldam_scale: float = 30.0,
    drw_start_epoch: int = 10,
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
    if image_transform not in {"default", "letterbox"}:
        raise ValueError("image_transform仅支持default或letterbox")
    if loss_strategy not in {"auto", "weighted_ce", "logit_adjusted", "ldam_drw"}:
        raise ValueError("loss_strategy仅支持auto、weighted_ce、logit_adjusted或ldam_drw")
    if logit_adjustment_tau < 0.0:
        raise ValueError("logit adjustment的tau不能小于0")
    if ldam_max_margin <= 0.0 or ldam_scale <= 0.0 or drw_start_epoch < 0:
        raise ValueError("LDAM-DRW参数不合法")

    config_dir = output_dir.parent / "ultralytics_config"
    matplotlib_dir = output_dir.parent / "matplotlib_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir.resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir.resolve()))
    # Some Windows environments report an incomplete CPU feature set to Polars although
    # its installed binary runs correctly. Ultralytics imports Polars only to serialize
    # results.csv into the checkpoint at the end of each epoch.
    if os.name == "nt":
        os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")
    if os.name == "nt" and "WINDIR" not in os.environ:
        os.environ["WINDIR"] = os.environ.get("SystemRoot", r"C:\Windows")

    from ultralytics import YOLO

    trainer = YOLO(model)
    trainer_class = None
    resolved_strategy = loss_strategy
    if resolved_strategy == "auto":
        resolved_strategy = "weighted_ce" if class_weight_power > 0.0 else "standard_ce"
    if resolved_strategy != "standard_ce":
        from water_agent.vision.long_tail import make_long_tail_classification_trainer

        trainer_class = make_long_tail_classification_trainer(
            strategy=resolved_strategy,
            power=class_weight_power,
            cap=class_weight_cap,
            logit_adjustment_tau=logit_adjustment_tau,
            ldam_max_margin=ldam_max_margin,
            ldam_scale=ldam_scale,
            drw_start_epoch=drw_start_epoch,
            label_smoothing=label_smoothing,
            image_transform=image_transform,
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
        "image_transform": image_transform,
        "loss_strategy": resolved_strategy,
        "logit_adjustment_tau": logit_adjustment_tau,
        "ldam_max_margin": ldam_max_margin,
        "ldam_scale": ldam_scale,
        "drw_start_epoch": drw_start_epoch,
        "note": "standard_ce为普通交叉熵；weighted_ce为有上限的逆频率幂次加权；logit_adjusted为训练期类别先验Logit调整；ldam_drw为大间隔长尾损失加延迟重加权。",
    }
