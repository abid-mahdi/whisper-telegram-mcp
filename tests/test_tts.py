"""Tests for the TTS module."""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Optional

from whisper_telegram_mcp.tts import (
    TTSResult,
    _map_to_openai_voice,
    _map_to_macos_voice,
    auto_tts,
)


@dataclass
class FakeConfig:
    tts_backend: str = "auto"
    tts_voice: str = "af_sky"
    kokoro_base_url: str = "http://127.0.0.1:8880/v1"
    openai_api_key: Optional[str] = None


class TestTTSResult:
    def test_success_result(self):
        r = TTSResult(file_path="/tmp/test.ogg", backend="kokoro", voice="af_sky", success=True)
        assert r.success is True
        assert r.file_path == "/tmp/test.ogg"
        assert r.error is None

    def test_failure_factory(self):
        r = TTSResult.failure("kokoro", "Connection refused")
        assert r.success is False
        assert r.file_path is None
        assert r.backend == "kokoro"
        assert r.error == "Connection refused"
        assert r.voice == ""

    def test_to_dict(self):
        r = TTSResult(file_path="/tmp/out.ogg", backend="openai", voice="nova", success=True)
        d = r.to_dict()
        assert d == {
            "file_path": "/tmp/out.ogg",
            "backend": "openai",
            "voice": "nova",
            "success": True,
            "error": None,
        }

    def test_failure_to_dict(self):
        r = TTSResult.failure("macos", "say not found")
        d = r.to_dict()
        assert d["success"] is False
        assert d["error"] == "say not found"
        assert d["file_path"] is None


class TestVoiceMapping:
    def test_kokoro_to_openai(self):
        assert _map_to_openai_voice("af_sky") == "nova"
        assert _map_to_openai_voice("am_adam") == "echo"
        assert _map_to_openai_voice("af_sarah") == "alloy"
        assert _map_to_openai_voice("am_michael") == "onyx"
        assert _map_to_openai_voice("af_bella") == "shimmer"

    def test_openai_passthrough(self):
        for v in ("alloy", "echo", "fable", "onyx", "nova", "shimmer"):
            assert _map_to_openai_voice(v) == v

    def test_unknown_voice_defaults_to_nova(self):
        assert _map_to_openai_voice("unknown_voice_xyz") == "nova"

    def test_kokoro_to_macos(self):
        assert _map_to_macos_voice("af_sky") == "Samantha"
        assert _map_to_macos_voice("am_adam") == "Daniel"
        assert _map_to_macos_voice("am_michael") == "Fred"

    def test_openai_to_macos(self):
        assert _map_to_macos_voice("alloy") == "Samantha"
        assert _map_to_macos_voice("echo") == "Fred"

    def test_unknown_voice_defaults_to_samantha(self):
        assert _map_to_macos_voice("unknown_voice_xyz") == "Samantha"


class TestAutoTTS:
    @pytest.mark.asyncio
    async def test_kokoro_available_uses_kokoro(self, tmp_path):
        cfg = FakeConfig(tts_backend="auto")
        out = str(tmp_path / "out.ogg")

        with patch("whisper_telegram_mcp.tts._is_kokoro_running", new_callable=AsyncMock, return_value=True), \
             patch("whisper_telegram_mcp.tts._kokoro_tts", new_callable=AsyncMock) as mock_kokoro:
            result = await auto_tts("Hello", cfg, output_path=out)

        assert result.success is True
        assert result.backend == "kokoro"
        assert result.voice == "af_sky"
        mock_kokoro.assert_awaited_once_with("Hello", "af_sky", cfg.kokoro_base_url, out)

    @pytest.mark.asyncio
    async def test_kokoro_down_falls_back_to_openai(self, tmp_path):
        cfg = FakeConfig(tts_backend="auto", openai_api_key="sk-test")
        out = str(tmp_path / "out.ogg")

        with patch("whisper_telegram_mcp.tts._is_kokoro_running", new_callable=AsyncMock, return_value=False), \
             patch("whisper_telegram_mcp.tts._start_kokoro", new_callable=AsyncMock, return_value=False), \
             patch("whisper_telegram_mcp.tts._openai_tts", new_callable=AsyncMock) as mock_openai:
            result = await auto_tts("Hello", cfg, output_path=out)

        assert result.success is True
        assert result.backend == "openai"
        assert result.voice == "nova"
        mock_openai.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_backends_returns_error(self, tmp_path):
        cfg = FakeConfig(tts_backend="auto", openai_api_key=None)
        out = str(tmp_path / "out.ogg")

        with patch("whisper_telegram_mcp.tts._is_kokoro_running", new_callable=AsyncMock, return_value=False), \
             patch("whisper_telegram_mcp.tts._start_kokoro", new_callable=AsyncMock, return_value=False), \
             patch("whisper_telegram_mcp.tts._macos_tts", new_callable=AsyncMock, side_effect=FileNotFoundError("say not found")):
            result = await auto_tts("Hello", cfg, output_path=out)

        assert result.success is False
        assert "No TTS backend available" in result.error

    @pytest.mark.asyncio
    async def test_explicit_kokoro_backend_not_running(self, tmp_path):
        cfg = FakeConfig(tts_backend="kokoro")
        out = str(tmp_path / "out.ogg")

        with patch("whisper_telegram_mcp.tts._is_kokoro_running", new_callable=AsyncMock, return_value=False):
            result = await auto_tts("Hello", cfg, output_path=out)

        assert result.success is False
        assert result.backend == "kokoro"
        assert "not running" in result.error.lower()

    @pytest.mark.asyncio
    async def test_explicit_openai_no_key(self, tmp_path):
        cfg = FakeConfig(tts_backend="openai", openai_api_key=None)
        out = str(tmp_path / "out.ogg")

        result = await auto_tts("Hello", cfg, output_path=out)

        assert result.success is False
        assert result.backend == "openai"
        assert "OPENAI_API_KEY" in result.error

    @pytest.mark.asyncio
    async def test_explicit_macos_backend(self, tmp_path):
        cfg = FakeConfig(tts_backend="macos")
        out = str(tmp_path / "out.ogg")

        with patch("whisper_telegram_mcp.tts._macos_tts", new_callable=AsyncMock) as mock_macos:
            result = await auto_tts("Hello", cfg, output_path=out)

        assert result.success is True
        assert result.backend == "macos"
        assert result.voice == "Samantha"
        mock_macos.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_auto_generates_temp_path_if_none(self):
        cfg = FakeConfig(tts_backend="kokoro")

        with patch("whisper_telegram_mcp.tts._is_kokoro_running", new_callable=AsyncMock, return_value=True), \
             patch("whisper_telegram_mcp.tts._kokoro_tts", new_callable=AsyncMock):
            result = await auto_tts("Hello", cfg, output_path=None)

        assert result.success is True
        assert result.file_path is not None
        assert result.file_path.endswith(".ogg")
        # Cleanup temp file
        try:
            os.unlink(result.file_path)
        except OSError:
            pass

    @pytest.mark.asyncio
    async def test_kokoro_failure_in_auto_falls_through(self, tmp_path):
        cfg = FakeConfig(tts_backend="auto", openai_api_key="sk-test")
        out = str(tmp_path / "out.ogg")

        with patch("whisper_telegram_mcp.tts._is_kokoro_running", new_callable=AsyncMock, return_value=True), \
             patch("whisper_telegram_mcp.tts._kokoro_tts", new_callable=AsyncMock, side_effect=RuntimeError("boom")), \
             patch("whisper_telegram_mcp.tts._openai_tts", new_callable=AsyncMock) as mock_openai:
            result = await auto_tts("Hello", cfg, output_path=out)

        assert result.success is True
        assert result.backend == "openai"
        mock_openai.assert_awaited_once()
