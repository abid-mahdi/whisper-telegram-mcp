"""Tests for the MCP server tool functions."""
import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from whisper_telegram_mcp.transcribe import TranscriptionResult


def make_success_result(**kwargs):
    defaults = dict(text="Hello", language="en", language_probability=0.99,
                    duration=1.0, segments=[], backend="local", success=True)
    defaults.update(kwargs)
    return TranscriptionResult(**defaults)


class TestTranscribeAudioTool:
    def test_tool_is_callable(self):
        from whisper_telegram_mcp.server import transcribe_audio
        assert callable(transcribe_audio)

    @pytest.mark.asyncio
    async def test_success(self, test_wav_file, monkeypatch):
        monkeypatch.delenv("WHISPER_BACKEND", raising=False)
        monkeypatch.delenv("WHISPER_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mock_result = make_success_result(text="Test transcription")
        with patch("whisper_telegram_mcp.server.auto_transcribe", return_value=mock_result):
            from whisper_telegram_mcp.server import transcribe_audio
            result = await transcribe_audio(file_path=test_wav_file)
        assert result["success"] is True
        assert result["text"] == "Test transcription"

    @pytest.mark.asyncio
    async def test_missing_file(self):
        from whisper_telegram_mcp.server import transcribe_audio
        result = await transcribe_audio(file_path="/does/not/exist.ogg")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_returns_json_serializable_dict(self, test_wav_file, monkeypatch):
        import json
        monkeypatch.delenv("WHISPER_BACKEND", raising=False)
        monkeypatch.delenv("WHISPER_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mock_result = make_success_result()
        with patch("whisper_telegram_mcp.server.auto_transcribe", return_value=mock_result):
            from whisper_telegram_mcp.server import transcribe_audio
            result = await transcribe_audio(file_path=test_wav_file)
        json.dumps(result)
        assert isinstance(result, dict)


class TestTranscribeTelegramVoiceTool:
    def test_tool_is_callable(self):
        from whisper_telegram_mcp.server import transcribe_telegram_voice
        assert callable(transcribe_telegram_voice)

    @pytest.mark.asyncio
    async def test_success(self, test_wav_file, monkeypatch):
        monkeypatch.delenv("WHISPER_BACKEND", raising=False)
        monkeypatch.delenv("WHISPER_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mock_result = make_success_result(text="Voice message", backend="local")
        with patch("whisper_telegram_mcp.server.download_voice_message", new_callable=AsyncMock) as mock_dl:
            mock_dl.return_value = test_wav_file
            with patch("whisper_telegram_mcp.server.auto_transcribe", return_value=mock_result):
                from whisper_telegram_mcp.server import transcribe_telegram_voice
                result = await transcribe_telegram_voice(file_id="AwACxxx", bot_token="123:ABC")
        assert result["success"] is True
        assert result["text"] == "Voice message"

    @pytest.mark.asyncio
    async def test_no_token_returns_error(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        from whisper_telegram_mcp.server import transcribe_telegram_voice
        result = await transcribe_telegram_voice(file_id="AwACxxx")
        assert result["success"] is False
        assert "token" in result["error"].lower() or "TELEGRAM_BOT_TOKEN" in result["error"]

    @pytest.mark.asyncio
    async def test_download_error_propagates(self):
        from whisper_telegram_mcp.telegram import TelegramDownloadError
        with patch("whisper_telegram_mcp.server.download_voice_message", new_callable=AsyncMock) as mock_dl:
            mock_dl.side_effect = TelegramDownloadError("file not found")
            from whisper_telegram_mcp.server import transcribe_telegram_voice
            result = await transcribe_telegram_voice(file_id="bad_id", bot_token="123:ABC")
        assert result["success"] is False
        assert "file not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_temp_file_cleaned_up(self, tmp_path, monkeypatch):
        """Verify temp file is deleted after transcription."""
        monkeypatch.delenv("WHISPER_BACKEND", raising=False)
        monkeypatch.delenv("WHISPER_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mock_result = make_success_result()
        temp_file = tmp_path / "voice.oga"
        temp_file.write_bytes(b"fake audio")

        with patch("whisper_telegram_mcp.server.download_voice_message", new_callable=AsyncMock) as mock_dl:
            mock_dl.return_value = str(temp_file)
            with patch("whisper_telegram_mcp.server.auto_transcribe", return_value=mock_result):
                from whisper_telegram_mcp.server import transcribe_telegram_voice
                await transcribe_telegram_voice(file_id="AwACxxx", bot_token="123:ABC")

        assert not temp_file.exists(), "Temp file should be deleted after transcription"


class TestListModelsTool:
    def test_tool_is_callable(self):
        from whisper_telegram_mcp.server import list_models
        assert callable(list_models)

    @pytest.mark.asyncio
    async def test_contains_expected_models(self, monkeypatch):
        monkeypatch.delenv("WHISPER_BACKEND", raising=False)
        monkeypatch.delenv("WHISPER_MODEL", raising=False)
        from whisper_telegram_mcp.server import list_models
        result = await list_models()
        assert "models" in result
        for model in ("tiny", "base", "small", "large-v3", "turbo"):
            assert model in result["models"]

    @pytest.mark.asyncio
    async def test_has_current_model(self, monkeypatch):
        monkeypatch.delenv("WHISPER_BACKEND", raising=False)
        monkeypatch.delenv("WHISPER_MODEL", raising=False)
        from whisper_telegram_mcp.server import list_models
        result = await list_models()
        assert "current" in result
        assert result["current"] == "base"


class TestCheckBackendsTool:
    def test_tool_is_callable(self):
        from whisper_telegram_mcp.server import check_backends
        assert callable(check_backends)

    @pytest.mark.asyncio
    async def test_structure(self, monkeypatch):
        monkeypatch.delenv("WHISPER_BACKEND", raising=False)
        monkeypatch.delenv("WHISPER_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from whisper_telegram_mcp.server import check_backends
        result = await check_backends()
        assert "local" in result
        assert "openai" in result
        assert isinstance(result["local"]["available"], bool)
        assert isinstance(result["openai"]["available"], bool)

    @pytest.mark.asyncio
    async def test_local_available(self, monkeypatch):
        """faster-whisper is installed so local should be available."""
        monkeypatch.delenv("WHISPER_BACKEND", raising=False)
        monkeypatch.delenv("WHISPER_MODEL", raising=False)
        from whisper_telegram_mcp.server import check_backends
        result = await check_backends()
        assert result["local"]["available"] is True

    @pytest.mark.asyncio
    async def test_openai_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("WHISPER_BACKEND", raising=False)
        monkeypatch.delenv("WHISPER_MODEL", raising=False)
        from whisper_telegram_mcp.server import check_backends
        result = await check_backends()
        assert result["openai"]["available"] is False
