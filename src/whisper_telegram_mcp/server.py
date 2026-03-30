"""MCP server exposing Whisper transcription tools."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any, Optional

# CRITICAL: All logging to stderr — stdout is reserved for MCP protocol (stdio transport)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

from mcp.server.fastmcp import FastMCP

from whisper_telegram_mcp.config import Config
from whisper_telegram_mcp.transcribe import LocalBackend, OpenAIBackend, auto_transcribe
from whisper_telegram_mcp.telegram import TelegramDownloadError, download_voice_message

mcp = FastMCP(
    "whisper-telegram-mcp",
    instructions=(
        "Transcribe audio files and Telegram voice messages using Whisper. "
        "Use transcribe_audio for local files, transcribe_telegram_voice for "
        "Telegram voice message file_ids. Call check_backends first to verify setup."
    ),
)


def _error_dict(backend: str, message: str) -> dict[str, Any]:
    """Return a consistent error dict."""
    return {
        "text": "", "language": "", "language_probability": 0.0,
        "duration": 0.0, "segments": [], "backend": backend,
        "success": False, "error": message,
    }


@mcp.tool()
async def transcribe_audio(
    file_path: str,
    language: Optional[str] = None,
    word_timestamps: bool = False,
) -> dict[str, Any]:
    """Transcribe an audio file to text using Whisper.

    Supports OGG (Telegram voice), WAV, MP3, FLAC, and most common audio formats.

    Args:
        file_path: Absolute path to the audio file to transcribe.
        language: Optional ISO-639-1 language code (e.g. 'en', 'fr'). None = auto-detect.
        word_timestamps: If True, include word-level timestamps in segments.

    Returns:
        dict with: text, language, language_probability, duration, segments, backend, success, error
    """
    if not os.path.exists(file_path):
        return _error_dict("none", f"File not found: {file_path}")

    cfg = Config()
    if language:
        cfg.language = language
    # Run synchronous (CPU-bound) transcription in a thread to avoid blocking the event loop
    result = await asyncio.to_thread(auto_transcribe, file_path, cfg, word_timestamps)
    return result.to_dict()


@mcp.tool()
async def transcribe_telegram_voice(
    file_id: str,
    bot_token: Optional[str] = None,
    language: Optional[str] = None,
    word_timestamps: bool = False,
) -> dict[str, Any]:
    """Download and transcribe a Telegram voice message.

    Downloads the voice message from Telegram, transcribes it, then deletes the temp file.

    Args:
        file_id: The file_id from a Telegram voice message (from the Message object).
        bot_token: Telegram bot token. Falls back to TELEGRAM_BOT_TOKEN env var.
        language: Optional ISO-639-1 language code. None = auto-detect.
        word_timestamps: Include word-level timestamps in segments.

    Returns:
        Same dict structure as transcribe_audio.
    """
    token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        return _error_dict(
            "none",
            "No bot token provided. Pass bot_token argument or set TELEGRAM_BOT_TOKEN env var. "
        "Setup guide: https://github.com/abid-mahdi/whisper-telegram-mcp#telegram-bot-setup"
        )

    try:
        local_path = await download_voice_message(token, file_id)
    except TelegramDownloadError as exc:
        return _error_dict("none", str(exc))
    except Exception as exc:
        logger.exception("Unexpected error downloading Telegram voice message")
        return _error_dict("none", f"Download failed: {exc}")

    try:
        cfg = Config()
        if language:
            cfg.language = language
        result = await asyncio.to_thread(auto_transcribe, local_path, cfg, word_timestamps)
        return result.to_dict()
    finally:
        try:
            os.unlink(local_path)
        except OSError:
            pass


@mcp.tool()
async def list_models() -> dict[str, Any]:
    """List available Whisper model sizes with performance characteristics.

    Configure the active model via the WHISPER_MODEL environment variable.
    Default is 'base' -- a good balance of speed and accuracy for voice messages.
    """
    model_info = {
        "tiny":      {"params": "39M",   "speed": "fastest",  "accuracy": "lowest",  "vram": "~1GB"},
        "tiny.en":   {"params": "39M",   "speed": "fastest",  "accuracy": "low",     "vram": "~1GB",  "note": "English only"},
        "base":      {"params": "74M",   "speed": "fast",     "accuracy": "good",    "vram": "~1GB"},
        "base.en":   {"params": "74M",   "speed": "fast",     "accuracy": "good",    "vram": "~1GB",  "note": "English only"},
        "small":     {"params": "244M",  "speed": "moderate", "accuracy": "better",  "vram": "~2GB"},
        "small.en":  {"params": "244M",  "speed": "moderate", "accuracy": "better",  "vram": "~2GB",  "note": "English only"},
        "medium":    {"params": "769M",  "speed": "slow",     "accuracy": "high",    "vram": "~5GB"},
        "medium.en": {"params": "769M",  "speed": "slow",     "accuracy": "high",    "vram": "~5GB",  "note": "English only"},
        "large-v1":  {"params": "1550M", "speed": "slowest",  "accuracy": "highest", "vram": "~10GB"},
        "large-v2":  {"params": "1550M", "speed": "slowest",  "accuracy": "highest", "vram": "~10GB"},
        "large-v3":  {"params": "1550M", "speed": "slowest",  "accuracy": "highest", "vram": "~10GB"},
        "turbo":     {"params": "~800M", "speed": "fast",     "accuracy": "high",    "vram": "~6GB",  "note": "Recommended for accuracy+speed"},
    }
    cfg = Config()
    return {
        "models": model_info,
        "current": cfg.model,
        "recommendation": "Use 'base' for most voice messages. Use 'turbo' for best accuracy.",
        "configure_via": "WHISPER_MODEL environment variable",
    }


@mcp.tool()
async def check_backends() -> dict[str, Any]:
    """Check which transcription backends are available and configured.

    Call this first to verify your setup before transcribing.
    """
    cfg = Config()
    local = LocalBackend(model_size=cfg.model)
    openai_b = OpenAIBackend(api_key=cfg.openai_api_key)

    return {
        "local": {
            "available": local.is_available(),
            "model": cfg.model,
            "description": "faster-whisper -- local inference (free, private, no API key)",
            "configure_via": "WHISPER_MODEL env var",
        },
        "openai": {
            "available": openai_b.is_available(),
            "description": "OpenAI Whisper API -- cloud ($0.006/min)",
            "configure_via": "OPENAI_API_KEY env var",
        },
        "current_backend": cfg.backend,
        "configure_backend_via": "WHISPER_BACKEND env var (auto|local|openai)",
        "telegram": {
            "token_configured": bool(cfg.telegram_bot_token),
            "configure_via": "TELEGRAM_BOT_TOKEN env var",
        },
    }


@mcp.tool()
async def speak_text(
    text: str,
    voice: str = "af_sky",
    output_path: Optional[str] = None,
) -> dict[str, Any]:
    """Convert text to speech and return an OGG/Opus audio file path.

    Plays as a native voice note in Telegram when sent as an attachment.

    TTS backends (in priority order):
    1. Kokoro (local, free, natural-sounding) -- auto-starts via `uvx kokoro-fastapi`
    2. OpenAI TTS (cloud, requires OPENAI_API_KEY, ~$0.015/1k chars)
    3. macOS say (Mac only fallback, sounds robotic)

    Configure via TTS_BACKEND env var: "auto" | "kokoro" | "openai" | "macos"

    Args:
        text: Text to synthesise.
        voice: Voice name. Kokoro voices: af_sky, af_bella, af_sarah, am_adam, am_michael,
               bf_emma, bm_george, bm_lewis. OpenAI voices: alloy, echo, fable, onyx, nova, shimmer.
               Configure default via TTS_VOICE env var.
        output_path: Optional absolute path for the output .ogg file.

    Returns:
        dict with: file_path (absolute .ogg path), backend, voice, success, error
    """
    if not text.strip():
        return {"success": False, "error": "text is empty", "file_path": None, "backend": "none", "voice": ""}

    try:
        from whisper_telegram_mcp.tts import auto_tts
        cfg = Config()
        if voice != "af_sky":
            cfg.tts_voice = voice
        result = await auto_tts(text, cfg, output_path=output_path)
        d = result.to_dict()
        d["size_bytes"] = os.path.getsize(result.file_path) if result.file_path and os.path.exists(result.file_path) else 0
        return d
    except Exception as exc:
        logger.exception("speak_text failed")
        return {"success": False, "error": str(exc), "file_path": None, "backend": "none", "voice": ""}


def main() -> None:
    """Entry point for the MCP server (stdio transport)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
