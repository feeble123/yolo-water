from __future__ import annotations

import json
import time
from typing import Any, Protocol

from langchain_openai import ChatOpenAI

from water_agent.config import Settings
from water_agent.schemas import ClassificationResult, LabelGuidance, TaskPlan


class ExplanationProvider(Protocol):
    def explain(
        self,
        prompt: str,
        result: ClassificationResult,
        guidance: LabelGuidance | None = None,
        report: bool = False,
    ) -> str: ...


class QwenProviderError(RuntimeError):
    """不包含服务端响应正文、凭据或请求载荷的安全错误。"""

    def __init__(self, message: str, error_kind: str) -> None:
        super().__init__(message)
        self.error_kind = error_kind


class ChatClient(Protocol):
    def invoke(self, messages: list[tuple[str, str]]) -> Any: ...


class QwenChatProvider:
    """赛事OpenAI兼容接口适配器；只发送问题和工具结构化结果。"""

    def __init__(self, settings: Settings, client: ChatClient | None = None) -> None:
        self.model = settings.qwen_model
        self.max_prompt_chars = settings.qwen_max_prompt_chars
        if client is not None:
            self.client = client
            return
        secret = settings.read_qwen_api_key()
        self.client: ChatClient = ChatOpenAI(
            model=settings.qwen_model,
            base_url=settings.qwen_base_url,
            api_key=secret.get_secret_value(),
            temperature=0.2,
            top_p=0.95,
            max_tokens=settings.qwen_max_tokens,
            timeout=settings.qwen_timeout_seconds,
            max_retries=settings.qwen_max_retries,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": settings.qwen_enable_thinking}
            },
        )

    def _invoke(self, messages: list[tuple[str, str]]) -> str:
        try:
            response = self.client.invoke(messages)
        except Exception as exc:
            error_kind = type(exc).__name__
            raise QwenProviderError(
                f"Qwen解释服务暂时不可用（{error_kind}）", error_kind
            ) from exc
        content = response.content
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = "".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ).strip()
        else:
            text = str(content).strip()
        if not text:
            raise QwenProviderError("Qwen解释服务返回了空内容（EmptyResponse）", "EmptyResponse")
        return text

    def plan_task(self, prompt: str, image_available: bool) -> TaskPlan:
        safe_prompt = prompt[: self.max_prompt_chars]
        messages = [
            (
                "system",
                (
                    "你是水域智能体的受控任务计划器。只输出一个JSON对象，"
                    "不使用Markdown代码块。intent只能是image_assessment、patrol_report、"
                    "result_reliability、label_guidance。tools必须严格匹配："
                    "image_assessment=[classify_water_image,compose_assessment]；"
                    "patrol_report=[classify_water_image,lookup_label_guidance,compose_report]；"
                    "result_reliability=[classify_water_image,assess_review_need,compose_assessment]；"
                    "label_guidance=[lookup_label_guidance]。"
                    "reason不超过80个汉字。不得输出标签、路径、命令或其他字段。"
                ),
            ),
            (
                "human",
                f"用户问题：{safe_prompt}\n是否已上传图片：{'是' if image_available else '否'}",
            ),
        ]
        text = self._invoke(messages)
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        try:
            payload = json.loads(text)
            payload["planner_mode"] = "qwen"
            return TaskPlan.model_validate(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            raise QwenProviderError("Qwen任务计划格式无效（InvalidPlan）", "InvalidPlan") from exc

    def explain(
        self,
        prompt: str,
        result: ClassificationResult,
        guidance: LabelGuidance | None = None,
        report: bool = False,
    ) -> str:
        tool_payload = result.model_dump_json(exclude={"latency_ms"})
        safe_prompt = prompt[: self.max_prompt_chars]
        guidance_payload = guidance.model_dump_json() if guidance else "无"
        task_style = "巡查报告" if report else "简洁分析"
        messages = [
            (
                "system",
                (
                    "你是水域综合异常识别助手。只能依据视觉工具JSON解释，"
                    "不得修改标签、伪造目标框或断言违法事实。"
                    "使用‘疑似’措辞；若requires_review为true，必须明确建议人工复核。"
                    f"当前输出类型为{task_style}。不要输出思考过程。"
                    "简洁分析不超过120个汉字；巡查报告不超过220个汉字。"
                ),
            ),
            (
                "human",
                f"用户问题：{safe_prompt}\n视觉工具结果：{tool_payload}\n类别知识：{guidance_payload}",
            ),
        ]
        return self._invoke(messages)

    def healthcheck(self) -> dict[str, str | float]:
        started = time.perf_counter()
        self._invoke(
            [
                ("system", "只回复 OK。"),
                ("human", "连接测试"),
            ]
        )
        return {
            "status": "ok",
            "model": self.model,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
