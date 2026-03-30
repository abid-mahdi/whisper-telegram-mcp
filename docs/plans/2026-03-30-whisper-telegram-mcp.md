# whisper-telegram-mcp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade open-source MCP server that transcribes Telegram voice messages (OGG/opus format) using OpenAI Whisper — both local (faster-whisper) and cloud (OpenAI API) backends supported.

**Architecture:** A Python `FastMCP` server exposing 4 tools: `transcribe_audio` (file path), `transcribe_telegram_voice` (Telegram file_id), `list_models`, and `check_backends`. Backend selection is automatic — prefers local faster-whisper, falls back to OpenAI API if key is set. All config via environment variables.

**Tech Stack:** Python 3.10+, `mcp[cli]` (FastMCP), `faster-whisper` (local Whisper via CTranslate2+PyAV), `openai` SDK, `httpx` (async Telegram download), `pytest` + `pytest-asyncio`, `hatchling` build

---

## File Structure

```
whisper-telegram-mcp/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Pytest matrix (ubuntu+macos, Python 3.10-3.12)
│       └── publish.yml             # PyPI publish on v* tags
├── src/
│   └── whisper_telegram_mcp/
│       ├── __init__.py             # Version constant
│       ├── __main__.py             # `python -m whisper_telegram_mcp` entry point
│       ├── config.py               # All env var config (WHISPER_BACKEND, WHISPER_MODEL, etc.)
│       ├── transcribe.py           # Core transcription: LocalBackend + OpenAIBackend + auto_transcribe()
│       ├── telegram.py             # Telegram file download helpers (async httpx)
│       └── server.py               # FastMCP instance, @mcp.tool() decorators
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Shared fixtures: tmp audio file, mock WhisperModel
│   ├── test_config.py              # Config loading, defaults, validation
│   ├── test_transcribe.py          # Unit tests (mocked) + integration test (tiny model)
│   ├── test_telegram.py            # Download helpers (mocked httpx)
│   └── test_server.py              # MCP tool registration, tool invocation
├── pyproject.toml                  # Build, deps, scripts, pytest config
├── README.md                       # Hero section, badges, quick start, tools ref
├── LICENSE                         # MIT
├── .gitignore                      # .venv, __pycache__, *.pyc, CLAUDE.md, RESEARCH.md, .env
├── .mcp.json                       # Claude Code one-click integration
└── server.json                     # MCP Registry manifest
```

---

## Task 1: Project scaffold (pyproject.toml, .gitignore, LICENSE, package skeleton)

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `src/whisper_telegram_mcp/__init__.py`
- Create: `src/whisper_telegram_mcp/__main__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "whisper-telegram-mcp"
version = "0.1.0"
description = "MCP server for transcribing Telegram voice messages using Whisper"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
keywords = ["mcp", "whisper", "telegram", "transcription", "voice", "claude"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Multimedia :: Sound/Audio :: Speech",
]
dependencies = [
    "mcp[cli]>=1.0.0",
    "faster-whisper>=1.0.0",
    "openai>=1.0.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "respx>=0.21.0",
]

[project.scripts]
whisper-telegram-mcp = "whisper_telegram_mcp.server:main"

[project.urls]
Homepage = "https://github.com/abid-mahdi/whisper-telegram-mcp"
Repository = "https://github.com/abid-mahdi/whisper-telegram-mcp"
Issues = "https://github.com/abid-mahdi/whisper-telegram-mcp/issues"

[tool.hatch.build.targets.wheel]
packages = ["src/whisper_telegram_mcp"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "--cov=whisper_telegram_mcp --cov-report=term-missing"

[tool.coverage.run]
source = ["src"]
```

- [ ] **Step 2: Create .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*.so
.Python
.venv/
venv/
env/
dist/
build/
*.egg-info/
.eggs/

# Testing
.pytest_cache/
.coverage
htmlcov/
coverage.xml

# Environment
.env
*.env

# Dev tools
.DS_Store
.idea/
.vscode/
*.swp

# Project-specific (not for public)
CLAUDE.md
RESEARCH.md
```

- [ ] **Step 3: Create MIT LICENSE**

```
MIT License

Copyright (c) 2026 Abid Mahdi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 4: Create package __init__.py**

```python
"""whisper-telegram-mcp: Transcribe Telegram voice messages with Whisper."""

__version__ = "0.1.0"
```

- [ ] **Step 5: Create __main__.py**

```python
"""Allow running as: python -m whisper_telegram_mcp"""
from whisper_telegram_mcp.server import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Install dev dependencies**

```bash
cd /Users/abidmahdi/Documents/dev/whisper-telegram-mcp
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

- [ ] **Step 7: Commit**

```bash
git init
git add pyproject.toml .gitignore LICENSE src/
git commit -m "chore: project scaffold with pyproject.toml and package skeleton"
```

---

## Task 2: Config module

**Files:**
- Create: `src/whisper_telegram_mcp/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_config.py
import os
import pytest
from whisper_telegram_mcp.config import Config


def test_defaults():
    """Config has sensible defaults when no env vars set."""
    c = Config()
    assert c.backend == "auto"
    assert c.model == "base"
    assert c.openai_api_key is None
    assert c.telegram_bot_token is None
    assert c.language is None


def test_from_env(monkeypatch):
    """Config reads from environment variables."""
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
    """Invalid backend raises ValueError."""
    monkeypatch.setenv("WHISPER_BACKEND", "invalid")
    with pytest.raises(ValueError, match="WHISPER_BACKEND"):
        Config()


def test_invalid_model(monkeypatch):
    """Invalid model raises ValueError."""
    monkeypatch.setenv("WHISPER_MODEL", "huge")
    with pytest.raises(ValueError, match="WHISPER_MODEL"):
        Config()


def test_valid_models():
    """All documented model names are accepted."""
    from whisper_telegram_mcp.config import VALID_MODELS
    assert "tiny" in VALID_MODELS
    assert "base" in VALID_MODELS
    assert "large-v3" in VALID_MODELS
```

- [ ] **Step 2: Run to confirm all fail**

```bash
pytest tests/test_config.py -v
```
Expected: All FAIL (ImportError or AttributeError)

- [ ] **Step 3: Implement config.py**

```python
# src/whisper_telegram_mcp/config.py
"""Configuration from environment variables."""

import os
from dataclasses import dataclass, field
from typing import Optional

VALID_BACKENDS = {"auto", "local", "openai"}
VALID_MODELS = {"tiny", "tiny.en", "base", "base.en", "small", "small.en",
                "medium", "medium.en", "large-v1", "large-v2", "large-v3", "turbo"}


@dataclass
class Config:
    """MCP server configuration loaded from environment variables.

    Environment variables:
        WHISPER_BACKEND: "auto" | "local" | "openai" (default: "auto")
            auto = try local first, fall back to openai if API key is set
        WHISPER_MODEL: Whisper model size (default: "base")
        OPENAI_API_KEY: OpenAI API key (required for openai backend)
        TELEGRAM_BOT_TOKEN: Telegram bot token (for direct file download)
        WHISPER_LANGUAGE: ISO-639-1 language code, e.g. "en" (default: auto-detect)
    """

    backend: str = field(default_factory=lambda: os.getenv("WHISPER_BACKEND", "auto"))
    model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "base"))
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    telegram_bot_token: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN")
    )
    language: Optional[str] = field(
        default_factory=lambda: os.getenv("WHISPER_LANGUAGE")
    )

    def __post_init__(self) -> None:
        if self.backend not in VALID_BACKENDS:
            raise ValueError(
                f"WHISPER_BACKEND must be one of {sorted(VALID_BACKENDS)}, "
                f"got '{self.backend}'"
            )
        if self.model not in VALID_MODELS:
            raise ValueError(
                f"WHISPER_MODEL must be one of {sorted(VALID_MODELS)}, "
                f"got '{self.model}'"
            )
```

- [ ] **Step 4: Run tests — all should pass**

```bash
pytest tests/test_config.py -v
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/whisper_telegram_mcp/config.py tests/test_config.py
git commit -m "feat: config module with env var validation"
```

---

## Task 3: Core transcription module

**Files:**
- Create: `src/whisper_telegram_mcp/transcribe.py`
- Create: `tests/conftest.py`
- Create: `tests/test_transcribe.py`

- [ ] **Step 1: Create conftest.py with shared fixtures**

```python
# tests/conftest.py
"""Shared test fixtures."""

import struct
import wave
import tempfile
import os
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def test_wav_file(tmp_path):
    """Create a minimal valid WAV file for testing (1 second of silence)."""
    wav_path = tmp_path / "test_audio.wav"
    with wave.open(str(wav_path), "w") as f:
        f.setnchannels(1)        # mono
        f.setsampwidth(2)        # 16-bit
        f.setframerate(16000)    # 16kHz
        f.writeframes(b"\x00\x00" * 16000)  # 1 second silence
    return str(wav_path)


@pytest.fixture
def mock_whisper_result():
    """A mock transcription result matching faster-whisper's Segment structure."""
    segment = MagicMock()
    segment.start = 0.0
    segment.end = 1.0
    segment.text = " Hello world"
    segment.words = []
    info = MagicMock()
    info.language = "en"
    info.language_probability = 0.99
    info.duration = 1.0
    return [segment], info


@pytest.fixture
def mock_whisper_model(mock_whisper_result):
    """A mock WhisperModel that returns canned transcription."""
    model = MagicMock()
    model.transcribe.return_value = mock_whisper_result
    return model
```

- [ ] **Step 2: Write failing tests for transcribe.py**

```python
# tests/test_transcribe.py
"""Tests for the transcription module."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
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
            text="Hello world",
            language="en",
            language_probability=0.99,
            duration=1.0,
            segments=[{"start": 0.0, "end": 1.0, "text": "Hello world"}],
            backend="local",
            success=True,
        )
        assert r.text == "Hello world"
        assert r.success is True
        assert r.error is None

    def test_error_result(self):
        r = TranscriptionResult.error("local", "File not found")
        assert r.success is False
        assert r.error == "File not found"
        assert r.text == ""


class TestLocalBackend:
    def test_transcribe_returns_result(self, test_wav_file, mock_whisper_model):
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
        """Model should not be loaded until first transcription."""
        with patch("whisper_telegram_mcp.transcribe.WhisperModel") as mock_cls:
            backend = LocalBackend(model_size="base")
            mock_cls.assert_not_called()

    def test_segments_are_serialized(self, test_wav_file, mock_whisper_model):
        """Segments generator must be fully consumed (list) before return."""
        with patch("whisper_telegram_mcp.transcribe.WhisperModel", return_value=mock_whisper_model):
            backend = LocalBackend(model_size="base")
            result = backend.transcribe(test_wav_file)
        assert isinstance(result.segments, list)


class TestOpenAIBackend:
    def test_transcribe_success(self, test_wav_file):
        mock_response = MagicMock()
        mock_response.text = "Hello from OpenAI"
        mock_response.language = "en"
        mock_response.duration = 2.5
        mock_response.segments = [{"start": 0.0, "end": 2.5, "text": "Hello from OpenAI"}]

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


class TestAutoTranscribe:
    def test_auto_uses_local_when_available(self, test_wav_file, mock_whisper_model):
        config = Config()
        config.backend = "auto"
        config.model = "base"
        with patch("whisper_telegram_mcp.transcribe.WhisperModel", return_value=mock_whisper_model):
            result = auto_transcribe(test_wav_file, config)
        assert result.backend == "local"
        assert result.success is True

    def test_auto_falls_back_to_openai(self, test_wav_file):
        config = Config()
        config.backend = "auto"
        config.openai_api_key = "sk-test"

        mock_response = MagicMock()
        mock_response.text = "Fallback result"
        mock_response.language = "en"
        mock_response.duration = 1.0
        mock_response.segments = []

        # Make local backend fail
        with patch("whisper_telegram_mcp.transcribe.WhisperModel", side_effect=Exception("No GPU")):
            with patch("whisper_telegram_mcp.transcribe.OpenAI") as mock_openai_cls:
                mock_client = MagicMock()
                mock_client.audio.transcriptions.create.return_value = mock_response
                mock_openai_cls.return_value = mock_client
                result = auto_transcribe(test_wav_file, config)

        assert result.backend == "openai"
        assert result.success is True

    def test_force_openai_backend(self, test_wav_file):
        config = Config()
        config.backend = "openai"
        config.openai_api_key = "sk-test"

        mock_response = MagicMock()
        mock_response.text = "OpenAI result"
        mock_response.language = "en"
        mock_response.duration = 1.0
        mock_response.segments = []

        with patch("whisper_telegram_mcp.transcribe.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.audio.transcriptions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client
            result = auto_transcribe(test_wav_file, config)

        assert result.backend == "openai"

    def test_no_backends_available_returns_error(self, test_wav_file):
        config = Config()
        config.backend = "openai"
        config.openai_api_key = None  # No key = openai unavailable
        result = auto_transcribe(test_wav_file, config)
        assert result.success is False
        assert "backend" in result.error.lower() or "available" in result.error.lower()


@pytest.mark.integration
class TestLocalBackendIntegration:
    """Integration tests that use the actual tiny Whisper model.

    These tests download the tiny model (~150MB) on first run.
    Run with: pytest -m integration
    """

    def test_transcribe_silence(self, test_wav_file):
        """Tiny model should handle silent audio without crashing."""
        from whisper_telegram_mcp.transcribe import LocalBackend
        backend = LocalBackend(model_size="tiny")
        result = backend.transcribe(test_wav_file)
        assert result.success is True
        assert isinstance(result.text, str)
        assert result.backend == "local"
```

- [ ] **Step 3: Run to confirm all fail**

```bash
pytest tests/test_transcribe.py -v --ignore-glob="*integration*" -m "not integration"
```
Expected: All FAIL (ImportError)

- [ ] **Step 4: Implement transcribe.py**

```python
# src/whisper_telegram_mcp/transcribe.py
"""Core transcription logic supporting local (faster-whisper) and OpenAI backends."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Import lazily to avoid startup cost
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore[assignment,misc]

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""

    text: str
    language: str
    language_probability: float
    duration: float
    segments: list[dict[str, Any]]
    backend: str
    success: bool
    error: Optional[str] = None

    @classmethod
    def error(cls, backend: str, message: str) -> "TranscriptionResult":
        """Create an error result."""
        return cls(
            text="",
            language="",
            language_probability=0.0,
            duration=0.0,
            segments=[],
            backend=backend,
            success=False,
            error=message,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "segments": self.segments,
            "backend": self.backend,
            "success": self.success,
            "error": self.error,
        }


class LocalBackend:
    """Transcription using faster-whisper (local, free, private)."""

    def __init__(self, model_size: str = "base") -> None:
        self.model_size = model_size
        self._model: Optional[Any] = None

    def _load_model(self) -> Any:
        if self._model is None:
            if WhisperModel is None:
                raise ImportError("faster-whisper is not installed")
            logger.info("Loading faster-whisper model '%s'...", self.model_size)
            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
            )
            logger.info("Model loaded.")
        return self._model

    def is_available(self) -> bool:
        return WhisperModel is not None

    def transcribe(
        self,
        file_path: str,
        language: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        if not os.path.exists(file_path):
            return TranscriptionResult.error("local", f"File not found: {file_path}")
        try:
            model = self._load_model()
            segments_gen, info = model.transcribe(
                file_path,
                language=language,
                beam_size=5,
                word_timestamps=word_timestamps,
                vad_filter=True,
            )
            # IMPORTANT: segments is a generator — must consume before returning
            segments = list(segments_gen)
            full_text = "".join(s.text for s in segments).strip()
            serialized = [
                {
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                    "text": s.text.strip(),
                }
                for s in segments
            ]
            return TranscriptionResult(
                text=full_text,
                language=info.language,
                language_probability=round(info.language_probability, 4),
                duration=round(info.duration, 3),
                segments=serialized,
                backend="local",
                success=True,
            )
        except Exception as exc:
            logger.exception("Local transcription failed")
            return TranscriptionResult.error("local", str(exc))


class OpenAIBackend:
    """Transcription using OpenAI Whisper API."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key

    def is_available(self) -> bool:
        return bool(self.api_key) and OpenAI is not None

    def transcribe(
        self,
        file_path: str,
        language: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        if not os.path.exists(file_path):
            return TranscriptionResult.error("openai", f"File not found: {file_path}")
        if not self.api_key:
            return TranscriptionResult.error(
                "openai", "OPENAI_API_KEY not set"
            )
        try:
            client = OpenAI(api_key=self.api_key)
            with open(file_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    language=language,
                    timestamp_granularities=(
                        ["segment", "word"] if word_timestamps else ["segment"]
                    ),
                )
            segments = [
                {
                    "start": round(s.get("start", 0), 3),
                    "end": round(s.get("end", 0), 3),
                    "text": s.get("text", "").strip(),
                }
                for s in (getattr(response, "segments", None) or [])
            ]
            return TranscriptionResult(
                text=(getattr(response, "text", "") or "").strip(),
                language=getattr(response, "language", ""),
                language_probability=1.0,  # API doesn't expose this
                duration=round(getattr(response, "duration", 0.0) or 0.0, 3),
                segments=segments,
                backend="openai",
                success=True,
            )
        except Exception as exc:
            logger.exception("OpenAI transcription failed")
            return TranscriptionResult.error("openai", str(exc))


def auto_transcribe(
    file_path: str,
    config: "Config",  # type: ignore[name-defined]  # noqa: F821
    word_timestamps: bool = False,
) -> TranscriptionResult:
    """Select and run the appropriate backend based on config.

    auto mode: try local first; if it fails AND openai key is available, use OpenAI.
    local mode: local only.
    openai mode: OpenAI only.
    """
    from whisper_telegram_mcp.config import Config  # avoid circular at module level

    local = LocalBackend(model_size=config.model)
    openai_b = OpenAIBackend(api_key=config.openai_api_key)

    if config.backend == "local":
        if not local.is_available():
            return TranscriptionResult.error(
                "local", "faster-whisper not installed"
            )
        return local.transcribe(file_path, language=config.language,
                                word_timestamps=word_timestamps)

    elif config.backend == "openai":
        if not openai_b.is_available():
            return TranscriptionResult.error(
                "openai", "OpenAI backend not available — set OPENAI_API_KEY"
            )
        return openai_b.transcribe(file_path, language=config.language,
                                   word_timestamps=word_timestamps)

    else:  # auto
        if local.is_available():
            result = local.transcribe(file_path, language=config.language,
                                      word_timestamps=word_timestamps)
            if result.success:
                return result
            logger.warning("Local backend failed: %s. Trying OpenAI...", result.error)

        if openai_b.is_available():
            return openai_b.transcribe(file_path, language=config.language,
                                       word_timestamps=word_timestamps)

        return TranscriptionResult.error(
            "auto", "No transcription backend available. "
            "Install faster-whisper or set OPENAI_API_KEY."
        )
```

- [ ] **Step 5: Run unit tests — should pass**

```bash
pytest tests/test_transcribe.py -v -m "not integration"
```
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/whisper_telegram_mcp/transcribe.py tests/conftest.py tests/test_transcribe.py
git commit -m "feat: transcription module with local and OpenAI backends"
```

---

## Task 4: Telegram download module

**Files:**
- Create: `src/whisper_telegram_mcp/telegram.py`
- Create: `tests/test_telegram.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_telegram.py
"""Tests for Telegram file download helpers."""

import pytest
import httpx
import respx
from whisper_telegram_mcp.telegram import (
    TelegramDownloadError,
    download_voice_message,
    get_file_path,
)


BOT_TOKEN = "123456789:ABCDefGhIJklmNOPQrst"
FILE_ID = "AwACAgIAAxkBAAIBpg"
FILE_PATH = "voice/file_1.oga"


@pytest.mark.asyncio
@respx.mock
async def test_download_success(tmp_path):
    """Successfully downloads and saves a voice message."""
    # Mock getFile endpoint
    respx.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getFile").mock(
        return_value=httpx.Response(200, json={
            "ok": True,
            "result": {"file_path": FILE_PATH, "file_id": FILE_ID, "file_size": 1024}
        })
    )
    # Mock file download endpoint
    respx.get(f"https://api.telegram.org/file/bot{BOT_TOKEN}/{FILE_PATH}").mock(
        return_value=httpx.Response(200, content=b"OGG_AUDIO_DATA")
    )

    output_path = await download_voice_message(
        bot_token=BOT_TOKEN,
        file_id=FILE_ID,
        output_dir=str(tmp_path),
    )
    assert output_path.endswith(".oga") or output_path.endswith(".ogg")
    with open(output_path, "rb") as f:
        assert f.read() == b"OGG_AUDIO_DATA"


@pytest.mark.asyncio
@respx.mock
async def test_download_invalid_file_id():
    """Raises TelegramDownloadError on invalid file_id."""
    respx.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getFile").mock(
        return_value=httpx.Response(200, json={"ok": False, "description": "Bad Request: file not found"})
    )

    with pytest.raises(TelegramDownloadError, match="file not found"):
        await download_voice_message(BOT_TOKEN, "INVALID_FILE_ID")


@pytest.mark.asyncio
@respx.mock
async def test_download_http_error():
    """Raises TelegramDownloadError on HTTP error."""
    respx.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getFile").mock(
        return_value=httpx.Response(401, json={"ok": False, "description": "Unauthorized"})
    )

    with pytest.raises(TelegramDownloadError):
        await download_voice_message(BOT_TOKEN, FILE_ID)


@pytest.mark.asyncio
@respx.mock
async def test_get_file_path():
    """get_file_path returns the file_path string."""
    respx.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getFile").mock(
        return_value=httpx.Response(200, json={
            "ok": True,
            "result": {"file_path": FILE_PATH}
        })
    )
    path = await get_file_path(BOT_TOKEN, FILE_ID)
    assert path == FILE_PATH
```

- [ ] **Step 2: Run to confirm all fail**

```bash
pytest tests/test_telegram.py -v
```
Expected: All FAIL (ImportError)

- [ ] **Step 3: Implement telegram.py**

```python
# src/whisper_telegram_mcp/telegram.py
"""Telegram Bot API helpers for downloading voice messages."""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import httpx

TELEGRAM_API = "https://api.telegram.org"


class TelegramDownloadError(Exception):
    """Raised when a Telegram file download fails."""


async def get_file_path(bot_token: str, file_id: str) -> str:
    """Resolve a Telegram file_id to a downloadable file_path.

    Args:
        bot_token: Telegram bot token.
        file_id: Telegram file ID from a voice message.

    Returns:
        The file_path string used to construct the download URL.

    Raises:
        TelegramDownloadError: If the API call fails.
    """
    url = f"{TELEGRAM_API}/bot{bot_token}/getFile"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params={"file_id": file_id})
    if response.status_code != 200:
        raise TelegramDownloadError(
            f"HTTP {response.status_code} from Telegram API: {response.text}"
        )
    data = response.json()
    if not data.get("ok"):
        raise TelegramDownloadError(
            f"Telegram API error: {data.get('description', 'Unknown error')}"
        )
    return data["result"]["file_path"]


async def download_voice_message(
    bot_token: str,
    file_id: str,
    output_dir: Optional[str] = None,
) -> str:
    """Download a Telegram voice message to a local file.

    Args:
        bot_token: Telegram bot token.
        file_id: The file_id from the Telegram voice message.
        output_dir: Directory to save the file. Defaults to system temp dir.

    Returns:
        Absolute path to the downloaded file.

    Raises:
        TelegramDownloadError: If the download fails.
    """
    file_path = await get_file_path(bot_token, file_id)
    download_url = f"{TELEGRAM_API}/file/bot{bot_token}/{file_path}"

    async with httpx.AsyncClient() as client:
        response = await client.get(download_url)

    if response.status_code != 200:
        raise TelegramDownloadError(
            f"Failed to download file: HTTP {response.status_code}"
        )

    # Preserve original extension (.oga or .ogg); default to .oga (Telegram standard)
    ext = os.path.splitext(file_path)[1] or ".oga"
    save_dir = output_dir or tempfile.gettempdir()
    os.makedirs(save_dir, exist_ok=True)

    # Create a named temp file with the correct extension
    fd, local_path = tempfile.mkstemp(suffix=ext, dir=save_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(response.content)
    except Exception:
        os.unlink(local_path)
        raise

    return local_path
```

- [ ] **Step 4: Run tests — all should pass**

```bash
pytest tests/test_telegram.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/whisper_telegram_mcp/telegram.py tests/test_telegram.py
git commit -m "feat: telegram download module with async httpx"
```

---

## Task 5: MCP server with tools

**Files:**
- Create: `src/whisper_telegram_mcp/server.py`
- Create: `tests/test_server.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_server.py
"""Tests for the MCP server tool definitions."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from whisper_telegram_mcp.transcribe import TranscriptionResult
from whisper_telegram_mcp.config import VALID_MODELS


# ---------------------------------------------------------------------------
# Import the module-level mcp instance and tool functions
# ---------------------------------------------------------------------------

def get_tools():
    """Get tools registered on the MCP server."""
    from whisper_telegram_mcp import server
    return server.mcp


class TestTranscribeAudioTool:
    def test_tool_exists(self):
        """transcribe_audio tool must be registered."""
        from whisper_telegram_mcp.server import transcribe_audio
        assert callable(transcribe_audio)

    @pytest.mark.asyncio
    async def test_transcribe_audio_success(self, test_wav_file):
        mock_result = TranscriptionResult(
            text="Test transcription",
            language="en",
            language_probability=0.99,
            duration=1.0,
            segments=[],
            backend="local",
            success=True,
        )
        with patch("whisper_telegram_mcp.server.auto_transcribe", return_value=mock_result):
            from whisper_telegram_mcp.server import transcribe_audio
            result = await transcribe_audio(file_path=test_wav_file)
        assert result["success"] is True
        assert result["text"] == "Test transcription"
        assert result["backend"] == "local"

    @pytest.mark.asyncio
    async def test_transcribe_audio_missing_file(self):
        from whisper_telegram_mcp.server import transcribe_audio
        result = await transcribe_audio(file_path="/does/not/exist.ogg")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_transcribe_audio_returns_dict(self, test_wav_file):
        mock_result = TranscriptionResult(
            text="Hello",
            language="en",
            language_probability=0.95,
            duration=0.5,
            segments=[],
            backend="local",
            success=True,
        )
        with patch("whisper_telegram_mcp.server.auto_transcribe", return_value=mock_result):
            from whisper_telegram_mcp.server import transcribe_audio
            result = await transcribe_audio(file_path=test_wav_file)
        # Must return a plain dict (JSON-serializable)
        assert isinstance(result, dict)
        assert "text" in result
        assert "language" in result
        assert "duration" in result
        assert "segments" in result


class TestTranscribeTelegramVoiceTool:
    def test_tool_exists(self):
        from whisper_telegram_mcp.server import transcribe_telegram_voice
        assert callable(transcribe_telegram_voice)

    @pytest.mark.asyncio
    async def test_transcribe_telegram_voice_success(self, tmp_path, test_wav_file):
        mock_result = TranscriptionResult(
            text="Voice message transcription",
            language="en",
            language_probability=0.98,
            duration=3.0,
            segments=[],
            backend="local",
            success=True,
        )
        with patch("whisper_telegram_mcp.server.download_voice_message", new_callable=AsyncMock) as mock_dl:
            mock_dl.return_value = test_wav_file
            with patch("whisper_telegram_mcp.server.auto_transcribe", return_value=mock_result):
                from whisper_telegram_mcp.server import transcribe_telegram_voice
                result = await transcribe_telegram_voice(
                    file_id="AwACxxx",
                    bot_token="123:ABC",
                )
        assert result["success"] is True
        assert result["text"] == "Voice message transcription"

    @pytest.mark.asyncio
    async def test_transcribe_telegram_voice_no_token(self):
        """Without bot_token, check env var; if missing, return helpful error."""
        import os
        with patch.dict(os.environ, {}, clear=True):
            # Remove TELEGRAM_BOT_TOKEN if set
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            from whisper_telegram_mcp.server import transcribe_telegram_voice
            result = await transcribe_telegram_voice(file_id="AwACxxx")
        assert result["success"] is False
        assert "token" in result["error"].lower() or "TELEGRAM_BOT_TOKEN" in result["error"]

    @pytest.mark.asyncio
    async def test_transcribe_telegram_download_error(self):
        from whisper_telegram_mcp.telegram import TelegramDownloadError
        with patch("whisper_telegram_mcp.server.download_voice_message", new_callable=AsyncMock) as mock_dl:
            mock_dl.side_effect = TelegramDownloadError("file not found")
            from whisper_telegram_mcp.server import transcribe_telegram_voice
            result = await transcribe_telegram_voice(
                file_id="bad_id",
                bot_token="123:ABC",
            )
        assert result["success"] is False
        assert "file not found" in result["error"].lower()


class TestListModelsTool:
    def test_tool_exists(self):
        from whisper_telegram_mcp.server import list_models
        assert callable(list_models)

    @pytest.mark.asyncio
    async def test_list_models_returns_all_models(self):
        from whisper_telegram_mcp.server import list_models
        result = await list_models()
        assert "models" in result
        assert "base" in result["models"]
        assert "tiny" in result["models"]
        assert "large-v3" in result["models"]


class TestCheckBackendsTool:
    def test_tool_exists(self):
        from whisper_telegram_mcp.server import check_backends
        assert callable(check_backends)

    @pytest.mark.asyncio
    async def test_check_backends_structure(self):
        from whisper_telegram_mcp.server import check_backends
        result = await check_backends()
        assert "local" in result
        assert "openai" in result
        assert isinstance(result["local"]["available"], bool)
        assert isinstance(result["openai"]["available"], bool)

    @pytest.mark.asyncio
    async def test_check_backends_local_available(self):
        """faster-whisper is installed so local should be available."""
        from whisper_telegram_mcp.server import check_backends
        result = await check_backends()
        assert result["local"]["available"] is True

    @pytest.mark.asyncio
    async def test_check_backends_openai_without_key(self):
        import os
        os.environ.pop("OPENAI_API_KEY", None)
        from whisper_telegram_mcp.server import check_backends
        result = await check_backends()
        assert result["openai"]["available"] is False
```

- [ ] **Step 2: Run to confirm all fail**

```bash
pytest tests/test_server.py -v
```
Expected: All FAIL (ImportError)

- [ ] **Step 3: Implement server.py**

```python
# src/whisper_telegram_mcp/server.py
"""MCP server exposing Whisper transcription tools."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Optional

# MCP stdio transport: ALL logging must go to stderr, not stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

from mcp.server.fastmcp import FastMCP

from whisper_telegram_mcp.config import Config, VALID_MODELS
from whisper_telegram_mcp.transcribe import LocalBackend, OpenAIBackend, auto_transcribe
from whisper_telegram_mcp.telegram import TelegramDownloadError, download_voice_message

mcp = FastMCP(
    "whisper-telegram-mcp",
    instructions=(
        "Transcribe audio files and Telegram voice messages using Whisper. "
        "Use transcribe_audio for local files, transcribe_telegram_voice for "
        "Telegram voice message file_ids. Check check_backends to see what's available."
    ),
)

_config = Config()


@mcp.tool()
async def transcribe_audio(
    file_path: str,
    language: Optional[str] = None,
    word_timestamps: bool = False,
) -> dict[str, Any]:
    """Transcribe an audio file to text using Whisper.

    Supports OGG (Telegram voice), WAV, MP3, FLAC, and most common formats.
    Uses the configured backend (local faster-whisper or OpenAI API).

    Args:
        file_path: Absolute path to the audio file to transcribe.
        language: Optional ISO-639-1 language code (e.g. 'en', 'fr').
                  Leave None for auto-detection.
        word_timestamps: If True, include word-level timestamps in segments.

    Returns:
        dict with keys: text, language, language_probability, duration,
        segments, backend, success, error (if failed).
    """
    if not os.path.exists(file_path):
        return {
            "success": False,
            "error": f"File not found: {file_path}",
            "text": "",
            "language": "",
            "language_probability": 0.0,
            "duration": 0.0,
            "segments": [],
            "backend": "none",
        }

    cfg = Config()
    if language:
        cfg.language = language
    result = auto_transcribe(file_path, cfg, word_timestamps=word_timestamps)
    return result.to_dict()


@mcp.tool()
async def transcribe_telegram_voice(
    file_id: str,
    bot_token: Optional[str] = None,
    language: Optional[str] = None,
    word_timestamps: bool = False,
) -> dict[str, Any]:
    """Download and transcribe a Telegram voice message.

    Downloads the voice message using the Telegram Bot API, then transcribes it.
    The bot_token can be passed directly or set via TELEGRAM_BOT_TOKEN env var.

    Args:
        file_id: The file_id from a Telegram voice message object.
        bot_token: Telegram bot token. Falls back to TELEGRAM_BOT_TOKEN env var.
        language: Optional ISO-639-1 language code. None = auto-detect.
        word_timestamps: Include word-level timestamps in output.

    Returns:
        Same dict structure as transcribe_audio.
    """
    token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        return {
            "success": False,
            "error": "No bot token provided. Pass bot_token argument or set TELEGRAM_BOT_TOKEN.",
            "text": "",
            "language": "",
            "language_probability": 0.0,
            "duration": 0.0,
            "segments": [],
            "backend": "none",
        }

    try:
        local_path = await download_voice_message(token, file_id)
    except TelegramDownloadError as exc:
        return {
            "success": False,
            "error": str(exc),
            "text": "",
            "language": "",
            "language_probability": 0.0,
            "duration": 0.0,
            "segments": [],
            "backend": "none",
        }

    try:
        cfg = Config()
        if language:
            cfg.language = language
        result = auto_transcribe(local_path, cfg, word_timestamps=word_timestamps)
        return result.to_dict()
    finally:
        # Clean up the temp file
        try:
            os.unlink(local_path)
        except OSError:
            pass


@mcp.tool()
async def list_models() -> dict[str, Any]:
    """List available Whisper model sizes.

    Returns model names with size and performance characteristics.
    Use 'base' for fast, accurate transcription of most voice messages.
    Use 'large-v3' for maximum accuracy (requires more RAM and time).
    Configure the active model via WHISPER_MODEL environment variable.
    """
    model_info = {
        "tiny": {"params": "39M", "speed": "fastest", "accuracy": "lowest"},
        "tiny.en": {"params": "39M", "speed": "fastest", "accuracy": "low (English only)"},
        "base": {"params": "74M", "speed": "fast", "accuracy": "good"},
        "base.en": {"params": "74M", "speed": "fast", "accuracy": "good (English only)"},
        "small": {"params": "244M", "speed": "moderate", "accuracy": "better"},
        "small.en": {"params": "244M", "speed": "moderate", "accuracy": "better (English only)"},
        "medium": {"params": "769M", "speed": "slow", "accuracy": "high"},
        "medium.en": {"params": "769M", "speed": "slow", "accuracy": "high (English only)"},
        "large-v1": {"params": "1550M", "speed": "slowest", "accuracy": "highest"},
        "large-v2": {"params": "1550M", "speed": "slowest", "accuracy": "highest"},
        "large-v3": {"params": "1550M", "speed": "slowest", "accuracy": "highest"},
        "turbo": {"params": "~800M", "speed": "fast", "accuracy": "high"},
    }
    cfg = Config()
    return {
        "models": model_info,
        "current": cfg.model,
        "configure_via": "WHISPER_MODEL environment variable",
    }


@mcp.tool()
async def check_backends() -> dict[str, Any]:
    """Check which transcription backends are available.

    Returns availability status for local (faster-whisper) and OpenAI backends.
    Use this to verify your setup before transcribing.
    """
    cfg = Config()
    local = LocalBackend(model_size=cfg.model)
    openai_b = OpenAIBackend(api_key=cfg.openai_api_key)

    result: dict[str, Any] = {
        "local": {
            "available": local.is_available(),
            "model": cfg.model,
            "description": "faster-whisper local inference (free, private)",
            "configure_via": "WHISPER_MODEL env var",
        },
        "openai": {
            "available": openai_b.is_available(),
            "description": "OpenAI Whisper API ($0.006/min)",
            "configure_via": "OPENAI_API_KEY env var",
        },
        "current_backend": cfg.backend,
        "configure_backend_via": "WHISPER_BACKEND env var (auto|local|openai)",
    }
    return result


def main() -> None:
    """Entry point for the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/ -v -m "not integration"
```
Expected: All PASS (config, transcribe, telegram, server tests)

- [ ] **Step 5: Commit**

```bash
git add src/whisper_telegram_mcp/server.py tests/test_server.py
git commit -m "feat: MCP server with transcribe_audio, transcribe_telegram_voice, list_models, check_backends tools"
```

---

## Task 6: GitHub integration files (.mcp.json, server.json, CI workflows)

**Files:**
- Create: `.mcp.json`
- Create: `server.json`
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/publish.yml`

- [ ] **Step 1: Create .mcp.json (Claude Code one-click integration)**

```json
{
  "mcpServers": {
    "whisper-telegram-mcp": {
      "command": "uvx",
      "args": ["whisper-telegram-mcp"],
      "env": {
        "WHISPER_MODEL": "base",
        "WHISPER_BACKEND": "auto"
      }
    }
  }
}
```

- [ ] **Step 2: Create server.json (MCP Registry manifest)**

```json
{
  "name": "whisper-telegram-mcp",
  "description": "Transcribe Telegram voice messages using Whisper — supports local (faster-whisper) and OpenAI API backends",
  "version": "0.1.0",
  "author": "Abid Mahdi",
  "license": "MIT",
  "homepage": "https://github.com/abid-mahdi/whisper-telegram-mcp",
  "repository": "https://github.com/abid-mahdi/whisper-telegram-mcp",
  "tools": [
    {
      "name": "transcribe_audio",
      "description": "Transcribe a local audio file (OGG, WAV, MP3, etc.) using Whisper"
    },
    {
      "name": "transcribe_telegram_voice",
      "description": "Download and transcribe a Telegram voice message by file_id"
    },
    {
      "name": "list_models",
      "description": "List available Whisper model sizes and their characteristics"
    },
    {
      "name": "check_backends",
      "description": "Check which transcription backends (local/OpenAI) are available"
    }
  ],
  "installation": {
    "npm": null,
    "pip": "pip install whisper-telegram-mcp",
    "uvx": "uvx whisper-telegram-mcp"
  }
}
```

- [ ] **Step 3: Create CI workflow**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    name: Test Python ${{ matrix.python-version }} on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Install dependencies
        run: uv pip install --system -e ".[dev]"

      - name: Run tests (unit only, no integration)
        run: pytest tests/ -v -m "not integration" --cov=whisper_telegram_mcp --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        if: matrix.python-version == '3.12' && matrix.os == 'ubuntu-latest'
        with:
          file: coverage.xml
```

- [ ] **Step 4: Create publish workflow**

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI

on:
  push:
    tags:
      - "v*"

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: release
    permissions:
      id-token: write  # Required for trusted publishing

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install build tools
        run: pip install build

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

- [ ] **Step 5: Commit**

```bash
git add .mcp.json server.json .github/
git commit -m "chore: add MCP registry manifest, Claude Code integration, CI/CD workflows"
```

---

## Task 7: CLAUDE.md and RESEARCH.md (gitignored internal docs)

**Files:**
- Create: `CLAUDE.md`
- Create: `RESEARCH.md`

- [ ] **Step 1: Create CLAUDE.md**

```markdown
# whisper-telegram-mcp — Developer Notes

## Architecture

- `config.py`: All env var handling. Config is instantiated fresh per request.
- `transcribe.py`: Two backend classes (LocalBackend, OpenAIBackend) + auto_transcribe()
- `telegram.py`: Async httpx download helpers
- `server.py`: FastMCP instance. All @mcp.tool() decorators here.

## Key Gotchas

- faster-whisper segments are a GENERATOR. Always `list(segments)` before returning.
- All logging MUST go to stderr (logging.basicConfig stream=sys.stderr). stdout is MCP protocol.
- LocalBackend loads model lazily on first call — first transcription is slow (~2-5s for base model).
- Model files cached at ~/.cache/huggingface/hub/

## Testing

```bash
# Unit tests (no model download)
pytest tests/ -m "not integration" -v

# Integration tests (downloads ~150MB tiny model on first run)
pytest tests/ -m integration -v

# Full test suite with coverage
pytest tests/ --cov=whisper_telegram_mcp
```

## Running Locally

```bash
# Install
pip install -e ".[dev]"

# Run server (connects via stdio to Claude Code)
python -m whisper_telegram_mcp

# Dev mode with MCP Inspector
uvx --refresh mcp dev src/whisper_telegram_mcp/server.py
```

## Adding a New Tool

1. Add function with `@mcp.tool()` decorator in `server.py`
2. Write test in `tests/test_server.py`
3. Update README tool table
```

- [ ] **Step 2: Create RESEARCH.md with findings summary**

Include key research findings from the research phase (see research notes).

- [ ] **Step 3: Verify both are in .gitignore**

```bash
git check-ignore -v CLAUDE.md RESEARCH.md
```
Expected: Both files shown as gitignored.

- [ ] **Step 4: No commit needed — these are gitignored**

---

## Task 8: README

**Files:**
- Create: `README.md`

The README must include:
- Project name + one-line tagline
- Badges: PyPI (placeholder), Python version, License, CI
- What it does (2-3 sentences)
- Quick start (one `uvx` command)
- Claude Code config snippet
- All 4 tools with descriptions
- Environment variables table
- Architecture diagram (ASCII)
- Contributing section
- License

Write the complete, polished README. This is what gets GitHub stars.

- [ ] **Step 1: Write README.md** (see content specification above)

- [ ] **Step 2: Verify it renders correctly**

```bash
# Check no broken markdown
python3 -c "import pathlib; print(len(pathlib.Path('README.md').read_text()), 'chars')"
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add comprehensive README with quick start, tools reference, and config guide"
```

---

## Task 9: Final verification and GitHub push

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v -m "not integration"
```
Expected: All tests PASS, coverage > 80%

- [ ] **Step 2: Verify package installs cleanly**

```bash
pip install -e . --quiet
python -c "import whisper_telegram_mcp; print(whisper_telegram_mcp.__version__)"
```
Expected: `0.1.0`

- [ ] **Step 3: Verify MCP server starts**

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | python -m whisper_telegram_mcp
```
Expected: JSON response with serverInfo containing "whisper-telegram-mcp"

- [ ] **Step 4: Create GitHub repo (public) and push**

```bash
gh repo create whisper-telegram-mcp --public --description "🎙️ MCP server for transcribing Telegram voice messages using Whisper" --push --source=.
```

- [ ] **Step 5: Add GitHub topics for discoverability**

```bash
gh repo edit --add-topic mcp,whisper,telegram,transcription,voice,claude,openai,faster-whisper
```

- [ ] **Step 6: Verify repo is public and looks correct**

```bash
gh repo view --web
```
