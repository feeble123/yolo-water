from __future__ import annotations

import argparse
import json
from pathlib import Path

from water_agent.agent.qwen_provider import QwenChatProvider, QwenProviderError
from water_agent.config import Settings
from water_agent.labels import WaterLabel
from water_agent.runtime import build_agent
from water_agent.schemas import ClassificationResult, ClassScore
from water_agent.tools.delivery import validate_delivery
from water_agent.tools.release import build_release
from water_agent.tools.submission import validate_submission
from water_agent.vision.batch_predict import predict_batch
from water_agent.vision.data_audit import audit_training_data, write_audit
from water_agent.vision.decision_policy import evaluate_oof_decision_policy
from water_agent.vision.evaluate import evaluate_model
from water_agent.vision.folds import prepare_grouped_folds, verify_fold_copies
from water_agent.vision.label_review import build_label_review_manifest
from water_agent.vision.oof import aggregate_oof
from water_agent.vision.train import run_training


def _validate(args: argparse.Namespace) -> int:
    result = validate_submission(args.result)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    return 0 if result.valid else 1


def _validate_delivery(args: argparse.Namespace) -> int:
    result = validate_delivery(args.root)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    return 0 if result.valid else 1


def _build_delivery(args: argparse.Namespace) -> int:
    try:
        result = build_release(
            project_root=args.project_root,
            result_path=args.result,
            output_path=args.output,
            confirm_real_result=args.confirm_real_result,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    return 0


def _audit_training(args: argparse.Namespace) -> int:
    report = audit_training_data(
        annotations_path=args.annotations,
        images_dir=args.images,
        near_duplicate_distance=args.near_duplicate_distance,
    )
    write_audit(report, args.output)
    summary = {
        "record_count": report["record_count"],
        "label_counts": report["label_counts"],
        "exact_duplicate_groups": len(report["exact_duplicate_groups"]),
        "near_duplicate_groups": len(report["near_duplicate_groups"]),
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _prepare_folds(args: argparse.Namespace) -> int:
    summary = prepare_grouped_folds(
        annotations_path=args.annotations,
        images_dir=args.images,
        audit_path=args.audit,
        output_dir=args.output,
        n_splits=args.folds,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _train(args: argparse.Namespace) -> int:
    result = run_training(
        data_dir=args.data,
        model=args.model,
        output_dir=args.output,
        run_name=args.name,
        epochs=args.epochs,
        image_size=args.image_size,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        class_weight_power=args.class_weight_power,
        class_weight_cap=args.class_weight_cap,
        label_smoothing=args.label_smoothing,
        image_transform=args.image_transform,
        loss_strategy=args.loss_strategy,
        logit_adjustment_tau=args.logit_adjustment_tau,
        ldam_max_margin=args.ldam_max_margin,
        ldam_scale=args.ldam_scale,
        drw_start_epoch=args.drw_start_epoch,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _verify_folds(args: argparse.Namespace) -> int:
    result = verify_fold_copies(images_dir=args.images, output_dir=args.folds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


def _evaluate(args: argparse.Namespace) -> int:
    result = evaluate_model(
        weights=args.weights,
        data_dir=args.data,
        output_dir=args.output,
        image_size=args.image_size,
        batch=args.batch,
        device=args.device,
        image_transform=args.image_transform,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _aggregate_oof(args: argparse.Namespace) -> int:
    result = aggregate_oof(args.evaluations, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _evaluate_decision_policy(args: argparse.Namespace) -> int:
    result = evaluate_oof_decision_policy(
        predictions_path=args.predictions,
        output_path=args.output,
        min_class_support=args.min_class_support,
        max_bias=args.max_bias,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "raw_metrics": result["raw_metrics"],
                "nested_biased_metrics": result["nested_biased_metrics"],
                "constraints": result["policy_constraints"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _build_label_review(args: argparse.Namespace) -> int:
    report = build_label_review_manifest(
        oof_predictions_path=args.predictions,
        training_images_dir=args.training_images,
        output_path=args.output,
        max_errors_per_label=args.max_errors_per_label,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "spreadsheet": report["spreadsheet_path"],
                "candidate_count": report["candidate_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _predict_batch(args: argparse.Namespace) -> int:
    result = predict_batch(
        input_dir=args.input,
        weights=args.weights,
        output=args.output,
        image_size=args.image_size,
        batch=args.batch,
        device=args.device,
        temperature=args.temperature,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _verify_qwen(args: argparse.Namespace) -> int:
    try:
        provider = QwenChatProvider(Settings())
        if args.explanation:
            started = __import__("time").perf_counter()
            answer = provider.explain(
                "请简要解释识别结果并说明是否需要人工复核。",
                ClassificationResult(
                    label=WaterLabel.NORMAL,
                    confidence=0.8,
                    top_k=[ClassScore(label=WaterLabel.NORMAL, probability=0.8)],
                    requires_review=False,
                    model_version="connection-test",
                    latency_ms=0,
                ),
            )
            result = {
                "status": "ok",
                "mode": "explanation",
                "model": provider.model,
                "latency_ms": round((__import__("time").perf_counter() - started) * 1000, 2),
                "answer_chars": len(answer),
            }
        else:
            result = provider.healthcheck()
    except (QwenProviderError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _analyze_image(args: argparse.Namespace) -> int:
    settings = Settings()
    response = build_agent(settings).analyze(args.image, args.prompt)
    payload = response.model_dump(mode="json")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _serve_ui(args: argparse.Namespace) -> int:
    from water_agent.ui import launch

    launch(server_name=args.host, server_port=args.port, inbrowser=args.open_browser)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="water-agent")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-submission", help="校验赛事结果JSON")
    validate.add_argument("--result", type=Path, required=True)
    validate.set_defaults(handler=_validate)
    delivery = commands.add_parser("validate-delivery", help="校验赛事提交目录结构与敏感文件")
    delivery.add_argument("--root", type=Path, required=True)
    delivery.set_defaults(handler=_validate_delivery)
    build = commands.add_parser("build-delivery", help="白名单组装并生成赛事.tar.gz提交包")
    build.add_argument("--project-root", type=Path, default=Path.cwd())
    build.add_argument("--result", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--confirm-real-result",
        action="store_true",
        help="确认result.json由参赛者对官方测试图片真实推理生成",
    )
    build.set_defaults(handler=_build_delivery)
    audit = commands.add_parser("audit-training", help="只读审计训练图片与标签")
    audit.add_argument("--annotations", type=Path, required=True)
    audit.add_argument("--images", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--near-duplicate-distance", type=int, default=6)
    audit.set_defaults(handler=_audit_training)
    folds = commands.add_parser("prepare-folds", help="按近重复组隔离生成YOLO分类四折目录")
    folds.add_argument("--annotations", type=Path, required=True)
    folds.add_argument("--images", type=Path, required=True)
    folds.add_argument("--audit", type=Path, required=True)
    folds.add_argument("--output", type=Path, required=True)
    folds.add_argument("--folds", type=int, default=4)
    folds.add_argument("--seed", type=int, default=20260820)
    folds.set_defaults(handler=_prepare_folds)
    verify = commands.add_parser("verify-folds", help="验证衍生折与原训练图片完全隔离")
    verify.add_argument("--images", type=Path, required=True)
    verify.add_argument("--folds", type=Path, required=True)
    verify.set_defaults(handler=_verify_folds)
    train = commands.add_parser("train", help="训练Ultralytics图像分类模型")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--model", default="yolo26n-cls.pt")
    train.add_argument("--output", type=Path, default=Path("artifacts/training"))
    train.add_argument("--name", required=True)
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--image-size", type=int, default=640)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--device", default="0")
    train.add_argument("--workers", type=int, default=2)
    train.add_argument("--seed", type=int, default=20260820)
    train.add_argument("--class-weight-power", type=float, default=0.0)
    train.add_argument("--class-weight-cap", type=float, default=8.0)
    train.add_argument("--label-smoothing", type=float, default=0.0)
    train.add_argument(
        "--loss-strategy",
        choices=("auto", "weighted_ce", "logit_adjusted", "ldam_drw"),
        default="auto",
        help="auto沿用既有行为；logit_adjusted只在训练期按类别先验调整logit",
    )
    train.add_argument("--logit-adjustment-tau", type=float, default=0.0)
    train.add_argument("--ldam-max-margin", type=float, default=0.5)
    train.add_argument("--ldam-scale", type=float, default=30.0)
    train.add_argument("--drw-start-epoch", type=int, default=10)
    train.add_argument(
        "--image-transform",
        choices=("default", "letterbox"),
        default="default",
        help="default使用Ultralytics裁剪；letterbox保留完整画幅",
    )
    train.set_defaults(handler=_train)
    evaluate = commands.add_parser("evaluate", help="评估分类权重并输出长尾指标")
    evaluate.add_argument("--weights", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, required=True, help="仅接受验证集目录")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--image-size", type=int, default=640)
    evaluate.add_argument("--batch", type=int, default=8)
    evaluate.add_argument("--device", default="0")
    evaluate.add_argument("--image-transform", choices=("default", "letterbox"), default="default")
    evaluate.set_defaults(handler=_evaluate)
    oof = commands.add_parser("aggregate-oof", help="汇总多折OOF指标并拟合温度校准")
    oof.add_argument("--evaluations", type=Path, nargs="+", required=True)
    oof.add_argument("--output", type=Path, required=True)
    oof.set_defaults(handler=_aggregate_oof)
    policy = commands.add_parser("evaluate-decision-policy", help="以嵌套OOF检验保守类别偏置")
    policy.add_argument("--predictions", type=Path, required=True)
    policy.add_argument("--output", type=Path, required=True)
    policy.add_argument("--min-class-support", type=int, default=20)
    policy.add_argument("--max-bias", type=float, default=0.75)
    policy.set_defaults(handler=_evaluate_decision_policy)
    review = commands.add_parser("build-label-review", help="从训练集OOF错例生成只读人工复核清单")
    review.add_argument("--predictions", type=Path, required=True)
    review.add_argument("--training-images", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--max-errors-per-label", type=int, default=4)
    review.set_defaults(handler=_build_label_review)
    predict = commands.add_parser("predict-batch", help="多模型概率集成并生成赛事提交JSON")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--weights", type=Path, nargs="+", required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--image-size", type=int, default=320)
    predict.add_argument("--batch", type=int, default=16)
    predict.add_argument("--device", default="0")
    predict.add_argument("--temperature", type=float, default=1.0)
    predict.set_defaults(handler=_predict_batch)
    verify_qwen = commands.add_parser("verify-qwen", help="安全验证官方Qwen接口连通性")
    verify_qwen.add_argument(
        "--explanation", action="store_true", help="验证结构化视觉结果解释能力"
    )
    verify_qwen.set_defaults(handler=_verify_qwen)
    analyze = commands.add_parser("analyze-image", help="运行单图智能体端到端分析")
    analyze.add_argument("--image", type=Path, required=True)
    analyze.add_argument("--prompt", default="请分析这张图片中的水域异常")
    analyze.add_argument("--output", type=Path)
    analyze.set_defaults(handler=_analyze_image)
    serve = commands.add_parser("serve-ui", help="启动Gradio演示界面")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
    serve.add_argument("--open-browser", action="store_true")
    serve.set_defaults(handler=_serve_ui)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
