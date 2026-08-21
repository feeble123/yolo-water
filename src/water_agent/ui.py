from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from water_agent.agent import WaterAnalysisAgent
from water_agent.config import Settings
from water_agent.runtime import build_agent


def analyze_for_display(
    agent: WaterAnalysisAgent, image_path: str | None, prompt: str
) -> tuple[str, str, str, str, str, str, str]:
    try:
        response = (
            agent.analyze(Path(image_path), prompt.strip() or "请分析这张图片中的水域异常")
            if image_path
            else agent.consult(prompt.strip() or "请说明有漂浮物的含义")
        )
    except (OSError, RuntimeError, ValueError):
        return (
            "分析失败，请检查图片、模型配置或服务状态。",
            "-",
            "-",
            "暂无数据",
            "暂无数据",
            "暂无数据",
            "暂无数据",
        )
    if response.plan is None:
        return "分析未生成任务计划。", "-", "-", "暂无数据", "暂无数据", "暂无数据", "暂无数据"
    if response.result:
        review = "是" if response.result.requires_review else "否"
        if response.result.review_reason:
            review += f"：{response.result.review_reason}"
        top_k_rows = [
            f"| {item.label} | {item.probability:.2%} |" for item in response.result.top_k
        ]
        top_k = "| 候选类别 | 概率 |\n|---|---:|\n" + "\n".join(top_k_rows)
        summary = f"{response.result.label}（{response.result.confidence:.2%}）"
    else:
        summary = "无需图片"
        review = "不适用"
        top_k = "本次是类别知识咨询，未调用YOLO视觉模型。"
    trace_rows = [
        f"| {item.tool} | {item.status} | {item.latency_ms:.2f} ms |"
        for item in response.trace
    ]
    trace = "| 步骤 | 状态 | 耗时 |\n|---|---|---:|\n" + "\n".join(trace_rows)
    tools = " → ".join(item.value for item in response.plan.tools)
    plan = (
        f"- **任务类型：** `{response.plan.intent.value}`\n"
        f"- **计划来源：** `{response.plan.planner_mode}`\n"
        f"- **执行步骤：** {tools}\n"
        f"- **任务理解：** {response.plan.reason}"
    )
    if response.guidance:
        guidance = (
            f"- **类别：** {response.guidance.label}\n"
            f"- **说明：** {response.guidance.definition}\n"
            f"- **易混淆点：** {response.guidance.common_confusions[0]}\n"
            f"- **建议核查：** {response.guidance.review_checks[0]}"
        )
    else:
        guidance = "本任务不需要调用类别知识工具。"
    return response.answer, summary, review, plan, guidance, top_k, trace


def create_demo(
    settings: Settings | None = None, agent: WaterAnalysisAgent | None = None
) -> gr.Blocks:
    runtime_settings = settings or Settings()
    runtime_agent = agent or build_agent(runtime_settings)
    vision_mode = (
        "演示假模型（不可用于赛事结果）"
        if runtime_settings.use_fake_model
        else (
            f"四折真实YOLO集成（图像尺寸{runtime_settings.vision_image_size}，"
            f"设备{runtime_settings.vision_device}）"
        )
    )
    qwen_mode = "已启用" if runtime_settings.use_qwen else "未启用（使用本地确定性说明）"
    with gr.Blocks(title="水域综合异常识别智能体") as demo:
        gr.Markdown(
            "# 水域综合异常识别智能体\n"
            "上传图片并描述任务。智能体先制定受控计划，再调用YOLO视觉、类别知识与报告工具；"
            "低置信度或验证证据不足时会提示人工复核。\n\n"
            f"**当前视觉模式：{vision_mode}；Qwen：{qwen_mode}。**"
        )
        with gr.Row():
            image = gr.Image(type="filepath", label="待分析图片")
            with gr.Column():
                prompt = gr.Textbox(
                    value="请生成这张图片的水域巡查报告，并说明是否需要人工复核。",
                    label="任务描述",
                    lines=3,
                )
                analyze = gr.Button("开始分析", variant="primary")
                answer = gr.Markdown("等待分析。")
        with gr.Row():
            prediction = gr.Textbox(label="分类结果", interactive=False)
            review = gr.Textbox(label="是否需要人工复核", interactive=False)
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 智能体任务计划")
                plan = gr.Markdown("暂无数据")
            with gr.Column():
                gr.Markdown("### 工具知识与核查依据")
                guidance = gr.Markdown("暂无数据")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Top-3候选")
                top_k = gr.Markdown("暂无数据")
            with gr.Column():
                gr.Markdown("### 智能体工具轨迹")
                trace = gr.Markdown("暂无数据")
        analyze.click(
            fn=lambda image_path, question: analyze_for_display(
                runtime_agent, image_path, question
            ),
            inputs=[image, prompt],
            outputs=[answer, prediction, review, plan, guidance, top_k, trace],
        )
    return demo


def launch(**kwargs: Any) -> None:
    create_demo().launch(**kwargs)
