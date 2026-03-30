"""Configuration from environment variables."""
import os
from dataclasses import dataclass, field
from typing import Optional

VALID_BACKENDS = {"auto", "local", "openai"}
VALID_MODELS = {
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v1", "large-v2", "large-v3", "turbo"
}
VALID_TTS_BACKENDS = {"auto", "kokoro", "openai", "macos"}


@dataclass
class Config:
    """MCP server configuration loaded from environment variables.

    Environment variables:
        WHISPER_BACKEND: "auto" | "local" | "openai" (default: "auto")
        WHISPER_MODEL: Whisper model size (default: "base")
        OPENAI_API_KEY: OpenAI API key (required for openai backend)
        TELEGRAM_BOT_TOKEN: Telegram bot token (for direct file download)
        WHISPER_LANGUAGE: ISO-639-1 language code e.g. "en" (default: auto-detect)
        TTS_BACKEND: "auto" | "kokoro" | "openai" | "macos" (default: "auto")
        TTS_VOICE: Voice name for TTS (default: "af_sky")
        KOKORO_BASE_URL: Kokoro FastAPI base URL (default: "http://127.0.0.1:8880/v1")
    """
    backend: str = field(default_factory=lambda: os.getenv("WHISPER_BACKEND", "auto"))
    model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "base"))
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    telegram_bot_token: Optional[str] = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN"))
    language: Optional[str] = field(default_factory=lambda: os.getenv("WHISPER_LANGUAGE"))
    tts_backend: str = field(default_factory=lambda: os.getenv("TTS_BACKEND", "auto"))
    tts_voice: str = field(default_factory=lambda: os.getenv("TTS_VOICE", "af_sky"))
    kokoro_base_url: str = field(default_factory=lambda: os.getenv("KOKORO_BASE_URL", "http://127.0.0.1:8880/v1"))

    def __post_init__(self) -> None:
        if self.backend not in VALID_BACKENDS:
            raise ValueError(f"WHISPER_BACKEND must be one of {sorted(VALID_BACKENDS)}, got '{self.backend}'")
        if self.model not in VALID_MODELS:
            raise ValueError(f"WHISPER_MODEL must be one of {sorted(VALID_MODELS)}, got '{self.model}'")
        if self.tts_backend not in VALID_TTS_BACKENDS:
            raise ValueError(f"TTS_BACKEND must be one of {sorted(VALID_TTS_BACKENDS)}, got '{self.tts_backend}'")
