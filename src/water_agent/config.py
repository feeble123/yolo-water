from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from water_agent.labels import WaterLabel


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WATER_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qwen_base_url: str = "https://www.zsjsry.top/v1"
    qwen_model: str = "qwen3.6-27b"
    qwen_api_key: SecretStr | None = None
    qwen_api_key_file: Path | None = None
    qwen_timeout_seconds: float = Field(default=30.0, gt=0.0, le=120.0)
    qwen_max_retries: int = Field(default=1, ge=0, le=5)
    qwen_max_prompt_chars: int = Field(default=2000, ge=1, le=20_000)
    qwen_max_tokens: int = Field(default=192, ge=64, le=1024)
    qwen_enable_thinking: bool = False
    model_weights: Path = Path("artifacts/models/best.pt")
    ensemble_weights: list[Path] = Field(default_factory=list)
    ensemble_temperature: float = Field(default=1.0, gt=0.0)
    vision_image_size: int = Field(default=320, gt=0, le=4096)
    vision_device: str = "0"
    review_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    force_review_labels: list[WaterLabel] = Field(
        default_factory=lambda: [WaterLabel.ILLEGAL_BUILDING, WaterLabel.NORMAL]
    )
    enable_audit_log: bool = True
    audit_db_path: Path = Path("artifacts/logs/agent_audit.sqlite3")
    use_fake_model: bool = True
    use_qwen: bool = False
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=40_000_000, gt=0)

    def read_qwen_api_key(self) -> SecretStr:
        if self.qwen_api_key and self.qwen_api_key.get_secret_value().strip():
            return self.qwen_api_key
        if self.qwen_api_key_file:
            value = self.qwen_api_key_file.read_text(encoding="utf-8").strip()
            if value:
                return SecretStr(value)
        raise RuntimeError("未配置Qwen API密钥")


@lru_cache
def get_settings() -> Settings:
    return Settings()
