"""Tests for config module."""
import pytest
from whisper_telegram_mcp.config import Config


def test_defaults(monkeypatch):
    monkeypatch.delenv("WHISPER_BACKEND", raising=False)
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("WHISPER_LANGUAGE", raising=False)
    c = Config()
    assert c.backend == "auto"
    assert c.model == "base"
    assert c.openai_api_key is None
    assert c.telegram_bot_token is None
    assert c.language is None


def test_from_env(monkeypatch):
    monkeypatch.setenv("WHISPER_BACKEND", "openai")
    monkeypatch.setenv("WHISPER_MODEL", "large-v3")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("WHISPER_LANGUAGE", "en")
    c = Config()
    assert c.backend == "openai"
    assert c.model == "large-v3"
    assert c.openai_api_key == "sk-test"
    assert c.telegram_bot_token == "123:ABC"
    assert c.language == "en"


def test_invalid_backend(monkeypatch):
    monkeypatch.setenv("WHISPER_BACKEND", "invalid")
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    with pytest.raises(ValueError, match="WHISPER_BACKEND"):
        Config()


def test_invalid_model(monkeypatch):
    monkeypatch.delenv("WHISPER_BACKEND", raising=False)
    monkeypatch.setenv("WHISPER_MODEL", "huge")
    with pytest.raises(ValueError, match="WHISPER_MODEL"):
        Config()


def test_valid_models():
    from whisper_telegram_mcp.config import VALID_MODELS
    assert "tiny" in VALID_MODELS
    assert "base" in VALID_MODELS
    assert "large-v3" in VALID_MODELS
    assert "turbo" in VALID_MODELS
