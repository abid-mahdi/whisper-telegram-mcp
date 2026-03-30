"""Shared test fixtures."""
import wave
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def test_wav_file(tmp_path):
    """Create a minimal valid WAV file (1 second of silence at 16kHz mono)."""
    wav_path = tmp_path / "test_audio.wav"
    with wave.open(str(wav_path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\x00\x00" * 16000)
    return str(wav_path)


@pytest.fixture
def mock_whisper_result():
    """faster-whisper result: (segments_list, info)"""
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
    """Mock WhisperModel returning canned result."""
    model = MagicMock()
    model.transcribe.return_value = mock_whisper_result
    return model
