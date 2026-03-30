"""Text-to-speech backends: Kokoro (local) -> OpenAI TTS -> macOS say (fallback)."""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    file_path: Optional[str]
    backend: str
    voice: str
    success: bool
    error: Optional[str] = None

    @classmethod
    def failure(cls, backend: str, message: str) -> "TTSResult":
        return cls(file_path=None, backend=backend, voice="", success=False, error=message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "backend": self.backend,
            "voice": self.voice,
            "success": self.success,
            "error": self.error,
        }


async def _kokoro_tts(text: str, voice: str, base_url: str, output_path: str) -> None:
    """Call Kokoro FastAPI (OpenAI-compatible endpoint) for TTS."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("openai package required for Kokoro TTS: pip install openai")

    client = AsyncOpenAI(api_key="not-needed", base_url=base_url)
    async with client.audio.speech.with_streaming_response.create(
        model="kokoro",
        voice=voice,
        input=text,
        response_format="opus",
    ) as response:
        content = await response.read()

    with open(output_path, "wb") as f:
        f.write(content)


async def _openai_tts(text: str, voice: str, api_key: str, output_path: str) -> None:
    """Call OpenAI TTS API."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("openai package required: pip install openai")

    client = AsyncOpenAI(api_key=api_key)
    async with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="opus",
    ) as response:
        content = await response.read()

    with open(output_path, "wb") as f:
        f.write(content)


async def _macos_tts(text: str, voice: str, output_path: str) -> None:
    """macOS built-in TTS via say + PyAV encoding to OGG/Opus. Mac only."""
    import av as _av

    aiff_fd, aiff_path = tempfile.mkstemp(suffix=".aiff")
    os.close(aiff_fd)
    try:
        await asyncio.to_thread(
            subprocess.run,
            ["say", "-v", voice, "-o", aiff_path, text],
            check=True, capture_output=True, timeout=30,
        )

        def _encode():
            with _av.open(aiff_path) as inp:
                with _av.open(output_path, "w", format="ogg") as out:
                    out_stream = out.add_stream("libopus", rate=48000)
                    out_stream.layout = "mono"
                    for frame in inp.decode(audio=0):
                        frame.pts = None
                        for pkt in out_stream.encode(frame):
                            out.mux(pkt)
                    for pkt in out_stream.encode(None):
                        out.mux(pkt)

        await asyncio.to_thread(_encode)
    finally:
        try:
            os.unlink(aiff_path)
        except OSError:
            pass


async def _is_kokoro_running(base_url: str) -> bool:
    """Check if Kokoro FastAPI is running at the given base URL."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(base_url.rstrip("/v1").rstrip("/") + "/health")
            return resp.status_code == 200
    except Exception:
        return False


async def _start_kokoro(base_url: str) -> bool:
    """Kokoro FastAPI must be started manually — it is not on PyPI.

    To run Kokoro locally:
      Docker (simplest):  docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest
      From source:        git clone https://github.com/remsky/Kokoro-FastAPI && cd Kokoro-FastAPI && ./start-cpu.sh

    This function always returns False so auto_tts falls through to the next backend.
    """
    logger.info(
        "Kokoro not running. Start it with: "
        "docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest"
    )
    return False


async def auto_tts(
    text: str,
    config: Any,
    output_path: Optional[str] = None,
) -> TTSResult:
    """Select and run the appropriate TTS backend.

    Priority: kokoro (local, free, natural) -> openai (cloud) -> macos (fallback, robotic)

    auto mode:
      1. Check if Kokoro is running; if not, try to start it
      2. If Kokoro fails, use OpenAI TTS if API key is set
      3. Last resort: macOS say (Mac only)
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".ogg")
        os.close(fd)

    voice = config.tts_voice
    backend = config.tts_backend

    if backend in ("auto", "kokoro"):
        kokoro_available = await _is_kokoro_running(config.kokoro_base_url)
        if not kokoro_available and backend == "auto":
            logger.info("Kokoro not running, attempting auto-start...")
            kokoro_available = await _start_kokoro(config.kokoro_base_url)

        if kokoro_available:
            try:
                await _kokoro_tts(text, voice, config.kokoro_base_url, output_path)
                return TTSResult(file_path=output_path, backend="kokoro", voice=voice, success=True)
            except Exception as exc:
                logger.warning("Kokoro TTS failed: %s", exc)
                if backend == "kokoro":
                    return TTSResult.failure("kokoro", str(exc))
        elif backend == "kokoro":
            return TTSResult.failure(
                "kokoro",
                "Kokoro not running. Start with: "
                "docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest"
            )

    if backend in ("auto", "openai"):
        if config.openai_api_key:
            openai_voice = _map_to_openai_voice(voice)
            try:
                await _openai_tts(text, openai_voice, config.openai_api_key, output_path)
                return TTSResult(file_path=output_path, backend="openai", voice=openai_voice, success=True)
            except Exception as exc:
                logger.warning("OpenAI TTS failed: %s", exc)
                if backend == "openai":
                    return TTSResult.failure("openai", str(exc))
        elif backend == "openai":
            return TTSResult.failure("openai", "OPENAI_API_KEY not set")

    if backend in ("auto", "macos"):
        macos_voice = _map_to_macos_voice(voice)
        try:
            await _macos_tts(text, macos_voice, output_path)
            return TTSResult(file_path=output_path, backend="macos", voice=macos_voice, success=True)
        except Exception as exc:
            if backend == "macos":
                return TTSResult.failure("macos", str(exc))
            logger.warning("macOS TTS failed: %s", exc)

    return TTSResult.failure(
        "auto",
        "No TTS backend available. Start Kokoro ("
        "docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest"
        ") or set OPENAI_API_KEY."
    )


def _map_to_openai_voice(voice: str) -> str:
    """Map Kokoro voice names to OpenAI TTS voice names."""
    mapping = {
        "af_sky": "nova",
        "af_bella": "shimmer",
        "af_sarah": "alloy",
        "af_nicole": "nova",
        "am_adam": "echo",
        "am_michael": "onyx",
        "bf_emma": "shimmer",
        "bf_isabella": "nova",
        "bm_george": "onyx",
        "bm_lewis": "echo",
    }
    openai_voices = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
    if voice in openai_voices:
        return voice
    return mapping.get(voice, "nova")


def _map_to_macos_voice(voice: str) -> str:
    """Map Kokoro/OpenAI voice names to macOS voice names."""
    mapping = {
        "af_sky": "Samantha",
        "af_bella": "Flo (English (US))",
        "af_sarah": "Karen",
        "af_nicole": "Moira",
        "am_adam": "Daniel",
        "am_michael": "Fred",
        "alloy": "Samantha",
        "echo": "Fred",
        "fable": "Daniel",
        "onyx": "Fred",
        "nova": "Samantha",
        "shimmer": "Flo (English (US))",
    }
    return mapping.get(voice, "Samantha")
