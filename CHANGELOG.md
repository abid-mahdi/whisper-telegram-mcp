# Changelog

All notable changes are documented here. Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] - 2026-03-30

### Added
- Two-way voice for Claude via Telegram: transcribe incoming voice messages, reply with synthesised speech
- Local Whisper transcription via faster-whisper (free, private, no API key)
- OpenAI Whisper API transcription (cloud fallback)
- Kokoro TTS voice synthesis — natural-sounding, free, runs locally via Docker
- OpenAI TTS fallback (cloud, all voices: alloy, echo, fable, onyx, nova, shimmer)
- macOS `say` last-resort fallback (Mac only)
- Auto backend selection: tries best available, falls back gracefully
- `transcribe_audio` — transcribe any local audio file (OGG, WAV, MP3, FLAC, etc.)
- `transcribe_telegram_voice` — download and transcribe a Telegram voice message by file_id
- `speak_text` — synthesise speech and save as OGG/Opus voice note
- `list_models` — list available Whisper model sizes with speed/accuracy info
- `check_backends` — check which backends are available and configured
- Multi-language support with automatic language detection
- Word-level timestamps (optional)
- 42 unit tests, CI on Python 3.10–3.12, Ubuntu + macOS
