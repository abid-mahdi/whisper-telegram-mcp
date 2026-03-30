"""Configuration from environment variables."""
import os
from dataclasses import dataclass, field
from typing import Optional

VALID_BACKENDS = {"auto", "local", "openai"}
VALID_MODELS = {
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v1", "large-v2", "large-v3", "turbo"
}


@dataclass
class Config:
    """MCP server configuration loaded from environment variables.

    Environment variables:
        WHISPER_BACKEND: "auto" | "local" | "openai" (default: "auto")
        WHISPER_MODEL: Whisper model size (default: "base")
        OPENAI_API_KEY: OpenAI API key (required for openai backend)
        TELEGRAM_BOT_TOKEN: Telegram bot token (for direct file download)
        WHISPER_LANGUAGE: ISO-639-1 language code e.g. "en" (default: auto-detect)
    """
    backend: str = field(default_factory=lambda: os.getenv("WHISPER_BACKEND", "auto"))
    model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "base"))
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    telegram_bot_token: Optional[str] = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN"))
    language: Optional[str] = field(default_factory=lambda: os.getenv("WHISPER_LANGUAGE"))

    def __post_init__(self) -> None:
        if self.backend not in VALID_BACKENDS:
            raise ValueError(f"WHISPER_BACKEND must be one of {sorted(VALID_BACKENDS)}, got '{self.backend}'")
        if self.model not in VALID_MODELS:
            raise ValueError(f"WHISPER_MODEL must be one of {sorted(VALID_MODELS)}, got '{self.model}'")
