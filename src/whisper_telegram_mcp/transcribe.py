"""Core transcription logic supporting local (faster-whisper) and OpenAI backends."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

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
    def from_error(cls, backend: str, message: str) -> "TranscriptionResult":
        return cls(
            text="", language="", language_probability=0.0, duration=0.0,
            segments=[], backend=backend, success=False, error=message,
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
                raise ImportError("faster-whisper is not installed. Run: pip install faster-whisper")
            logger.info("Loading faster-whisper model '%s'...", self.model_size)
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            logger.info("Model '%s' loaded.", self.model_size)
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
            return TranscriptionResult.from_error("local", f"File not found: {file_path}")
        try:
            model = self._load_model()
            # IMPORTANT: transcribe() returns a generator for segments — must list() it
            segments_gen, info = model.transcribe(
                file_path,
                language=language,
                beam_size=5,
                word_timestamps=word_timestamps,
                vad_filter=True,
            )
            segments = list(segments_gen)  # consume generator NOW
            full_text = "".join(s.text for s in segments).strip()
            serialized = [
                {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
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
            return TranscriptionResult.from_error("local", str(exc))


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
            return TranscriptionResult.from_error("openai", f"File not found: {file_path}")
        if not self.api_key:
            return TranscriptionResult.from_error("openai", "OPENAI_API_KEY not set")
        try:
            client = OpenAI(api_key=self.api_key)
            with open(file_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    language=language,
                    timestamp_granularities=(["segment", "word"] if word_timestamps else ["segment"]),
                )
            # OpenAI SDK may return segment objects (Pydantic) not dicts — use getattr
            raw_segments = getattr(response, "segments", None) or []
            segments = [
                {
                    "start": round(float(getattr(s, "start", 0) or 0), 3),
                    "end": round(float(getattr(s, "end", 0) or 0), 3),
                    "text": str(getattr(s, "text", "") or "").strip(),
                }
                for s in raw_segments
            ]
            return TranscriptionResult(
                text=str(getattr(response, "text", "") or "").strip(),
                language=str(getattr(response, "language", "") or ""),
                language_probability=1.0,
                duration=round(float(getattr(response, "duration", 0.0) or 0.0), 3),
                segments=segments,
                backend="openai",
                success=True,
            )
        except Exception as exc:
            logger.exception("OpenAI transcription failed")
            return TranscriptionResult.from_error("openai", str(exc))


def auto_transcribe(
    file_path: str,
    config: Any,
    word_timestamps: bool = False,
) -> TranscriptionResult:
    """Select and run the appropriate backend based on config."""
    local = LocalBackend(model_size=config.model)
    openai_b = OpenAIBackend(api_key=config.openai_api_key)

    if config.backend == "local":
        if not local.is_available():
            return TranscriptionResult.from_error("local", "faster-whisper not installed")
        return local.transcribe(file_path, language=config.language, word_timestamps=word_timestamps)

    elif config.backend == "openai":
        if not openai_b.is_available():
            return TranscriptionResult.from_error("openai", "OpenAI backend not available — set OPENAI_API_KEY")
        return openai_b.transcribe(file_path, language=config.language, word_timestamps=word_timestamps)

    else:  # auto
        if local.is_available():
            result = local.transcribe(file_path, language=config.language, word_timestamps=word_timestamps)
            if result.success:
                return result
            logger.warning("Local backend failed: %s. Trying OpenAI...", result.error)

        if openai_b.is_available():
            return openai_b.transcribe(file_path, language=config.language, word_timestamps=word_timestamps)

        return TranscriptionResult.from_error(
            "auto",
            "No transcription backend available. Install faster-whisper or set OPENAI_API_KEY."
        )
