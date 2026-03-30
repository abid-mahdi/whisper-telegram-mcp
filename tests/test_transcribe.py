"""Tests for transcription module."""
import pytest
from unittest.mock import patch, MagicMock
from whisper_telegram_mcp.transcribe import (
    TranscriptionResult,
    LocalBackend,
    OpenAIBackend,
    auto_transcribe,
)
from whisper_telegram_mcp.config import Config


class TestTranscriptionResult:
    def test_success_result(self):
        r = TranscriptionResult(
            text="Hello world", language="en", language_probability=0.99,
            duration=1.0, segments=[{"start": 0.0, "end": 1.0, "text": "Hello world"}],
            backend="local", success=True,
        )
        assert r.text == "Hello world"
        assert r.success is True
        assert r.error is None

    def test_error_factory(self):
        r = TranscriptionResult.from_error("local", "File not found")
        assert r.success is False
        assert r.error == "File not found"
        assert r.text == ""

    def test_to_dict_keys(self):
        r = TranscriptionResult.from_error("local", "oops")
        d = r.to_dict()
        for key in ("text", "language", "language_probability", "duration", "segments", "backend", "success", "error"):
            assert key in d


class TestLocalBackend:
    def test_transcribe_success(self, test_wav_file, mock_whisper_model):
        with patch("whisper_telegram_mcp.transcribe.WhisperModel", return_value=mock_whisper_model):
            backend = LocalBackend(model_size="base")
            result = backend.transcribe(test_wav_file)
        assert result.success is True
        assert "Hello world" in result.text
        assert result.backend == "local"
        assert result.language == "en"

    def test_transcribe_missing_file(self):
        with patch("whisper_telegram_mcp.transcribe.WhisperModel"):
            backend = LocalBackend(model_size="base")
            result = backend.transcribe("/nonexistent/file.ogg")
        assert result.success is False
        assert result.error is not None

    def test_is_available_true(self):
        with patch("whisper_telegram_mcp.transcribe.WhisperModel"):
            backend = LocalBackend(model_size="base")
            assert backend.is_available() is True

    def test_model_loaded_lazily(self):
        with patch("whisper_telegram_mcp.transcribe.WhisperModel") as mock_cls:
            LocalBackend(model_size="base")
            mock_cls.assert_not_called()

    def test_segments_are_list(self, test_wav_file, mock_whisper_model):
        """Generator must be consumed -- segments must be a list."""
        with patch("whisper_telegram_mcp.transcribe.WhisperModel", return_value=mock_whisper_model):
            backend = LocalBackend(model_size="base")
            result = backend.transcribe(test_wav_file)
        assert isinstance(result.segments, list)


class TestOpenAIBackend:
    def _mock_openai_response(self, text="Hello from OpenAI", language="en", duration=2.5, with_segments=False):
        resp = MagicMock()
        resp.text = text
        resp.language = language
        resp.duration = duration
        if with_segments:
            seg = MagicMock()
            seg.start = 0.0
            seg.end = 2.5
            seg.text = text
            resp.segments = [seg]
        else:
            resp.segments = []
        return resp

    def test_transcribe_success(self, test_wav_file):
        mock_response = self._mock_openai_response()
        with patch("whisper_telegram_mcp.transcribe.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.audio.transcriptions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client
            backend = OpenAIBackend(api_key="sk-test")
            result = backend.transcribe(test_wav_file)
        assert result.success is True
        assert result.text == "Hello from OpenAI"
        assert result.backend == "openai"

    def test_is_available_with_key(self):
        backend = OpenAIBackend(api_key="sk-test")
        assert backend.is_available() is True

    def test_is_available_without_key(self):
        backend = OpenAIBackend(api_key=None)
        assert backend.is_available() is False

    def test_missing_file(self):
        backend = OpenAIBackend(api_key="sk-test")
        result = backend.transcribe("/nonexistent/file.ogg")
        assert result.success is False

    def test_no_api_key_returns_error(self, test_wav_file):
        backend = OpenAIBackend(api_key=None)
        result = backend.transcribe(test_wav_file)
        assert result.success is False
        assert "OPENAI_API_KEY" in result.error


class TestAutoTranscribe:
    def test_auto_uses_local_when_available(self, test_wav_file, mock_whisper_model, monkeypatch):
        monkeypatch.delenv("WHISPER_BACKEND", raising=False)
        monkeypatch.delenv("WHISPER_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = Config()
        with patch("whisper_telegram_mcp.transcribe.WhisperModel", return_value=mock_whisper_model):
            result = auto_transcribe(test_wav_file, cfg)
        assert result.backend == "local"
        assert result.success is True

    def test_auto_falls_back_to_openai_on_local_failure(self, test_wav_file, monkeypatch):
        monkeypatch.delenv("WHISPER_BACKEND", raising=False)
        monkeypatch.delenv("WHISPER_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = Config()
        cfg.openai_api_key = "sk-test"

        mock_response = MagicMock()
        mock_response.text = "Fallback result"
        mock_response.language = "en"
        mock_response.duration = 1.0
        mock_response.segments = []

        with patch("whisper_telegram_mcp.transcribe.WhisperModel", side_effect=Exception("No GPU")):
            with patch("whisper_telegram_mcp.transcribe.OpenAI") as mock_openai_cls:
                mock_client = MagicMock()
                mock_client.audio.transcriptions.create.return_value = mock_response
                mock_openai_cls.return_value = mock_client
                result = auto_transcribe(test_wav_file, cfg)

        assert result.backend == "openai"
        assert result.success is True

    def test_force_openai_backend(self, test_wav_file, monkeypatch):
        monkeypatch.delenv("WHISPER_BACKEND", raising=False)
        monkeypatch.delenv("WHISPER_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = Config()
        cfg.backend = "openai"
        cfg.openai_api_key = "sk-test"

        mock_response = MagicMock()
        mock_response.text = "OpenAI result"
        mock_response.language = "en"
        mock_response.duration = 1.0
        mock_response.segments = []

        with patch("whisper_telegram_mcp.transcribe.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.audio.transcriptions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client
            result = auto_transcribe(test_wav_file, cfg)

        assert result.backend == "openai"

    def test_no_backends_available_returns_error(self, test_wav_file, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("WHISPER_BACKEND", raising=False)
        monkeypatch.delenv("WHISPER_MODEL", raising=False)
        cfg = Config()
        cfg.backend = "openai"
        cfg.openai_api_key = None
        result = auto_transcribe(test_wav_file, cfg)
        assert result.success is False
        assert result.error is not None


@pytest.mark.integration
class TestLocalBackendIntegration:
    """Integration tests using the actual tiny Whisper model (~150MB download on first run).

    Run with: pytest -m integration
    """

    def test_transcribe_silence_does_not_crash(self, test_wav_file):
        """Tiny model should handle silent audio without crashing."""
        backend = LocalBackend(model_size="tiny")
        result = backend.transcribe(test_wav_file)
        assert result.success is True
        assert isinstance(result.text, str)
        assert result.backend == "local"
