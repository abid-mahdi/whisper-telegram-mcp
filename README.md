# whisper-telegram-mcp

> Transcribe and speak — two-way voice for Claude via Telegram

[![CI](https://github.com/abid-mahdi/whisper-telegram-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/abid-mahdi/whisper-telegram-mcp/actions)
[![Python Version](https://img.shields.io/pypi/pyversions/whisper-telegram-mcp)](https://pypi.org/project/whisper-telegram-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-compatible-brightgreen)](https://modelcontextprotocol.io)

An [MCP](https://modelcontextprotocol.io) server that gives Claude two-way voice capabilities via Telegram: transcribe incoming voice messages with Whisper, and reply with synthesized speech. Works with Claude Desktop, Claude Code, and any MCP-compatible client.

## What It Does

- **Transcribe local audio files** -- OGG, WAV, MP3, FLAC, and more
- **Transcribe Telegram voice messages** -- pass a `file_id`, get text back
- **Speak text as voice notes** -- synthesise speech and send back as OGG (plays as a voice note in Telegram)
- **Two transcription backends** -- local [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (free, private) or OpenAI Whisper API (cloud)
- **Auto mode** -- tries local first, falls back to OpenAI if it fails
- **Language detection** -- automatic or specify an ISO-639-1 code
- **Word-level timestamps** -- optional fine-grained timing

## Quick Start

### One command with `uvx`

```bash
uvx whisper-telegram-mcp
```

No installation needed -- `uvx` handles everything.

### Or install with pip

```bash
pip install whisper-telegram-mcp
whisper-telegram-mcp
```

## Integration

### Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

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

### Claude Code

Add to your project's `.mcp.json`:

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

## Tools

| Tool | Description |
|------|-------------|
| `transcribe_audio` | Transcribe a local audio file (OGG, WAV, MP3, etc.) to text |
| `transcribe_telegram_voice` | Download and transcribe a Telegram voice message by `file_id` |
| `speak_text` | Convert text to speech → OGG/Opus file (plays as voice note in Telegram) |
| `list_models` | List available Whisper model sizes with speed/accuracy info |
| `check_backends` | Check which backends (local/OpenAI) are available and configured |

### `transcribe_audio`

```
file_path: str        # Absolute path to audio file
language: str | None  # ISO-639-1 code (e.g. "en"), None = auto-detect
word_timestamps: bool # Include word-level timestamps (default: false)
```

### `transcribe_telegram_voice`

```
file_id: str          # Telegram voice message file_id
bot_token: str | None # Bot token (falls back to TELEGRAM_BOT_TOKEN env var)
language: str | None  # ISO-639-1 code, None = auto-detect
word_timestamps: bool # Include word-level timestamps (default: false)
```

### `speak_text`

Converts text to an OGG/Opus audio file using macOS built-in TTS. No API key required — completely free. Uses PyAV (bundled with faster-whisper) for encoding, so no system-level ffmpeg needed.

```
text: str             # Text to synthesise
voice: str            # macOS voice name (default: "Samantha")
output_path: str|None # Optional path for output .ogg file
```

**Available English voices:**

| Voice | Accent | Style |
|-------|--------|-------|
| `Samantha` | US | Natural female (default) |
| `Flo (English (US))` | US | Natural female |
| `Daniel` | British | Natural male |
| `Karen` | Australian | Natural female |
| `Moira` | Irish | Natural female |
| `Fred` | US | Classic male |

Run `say -v "?"` in your terminal to see all voices available on your system.

**Returns:**
```json
{
  "file_path": "/tmp/tmpXXX.ogg",
  "size_bytes": 16555,
  "voice": "Samantha",
  "success": true,
  "error": null
}
```

Send the returned `file_path` as a Telegram attachment and it will appear as a native voice note.

### Transcription response format

All transcription tools return:

```json
{
  "text": "Hello, this is a voice message.",
  "language": "en",
  "language_probability": 0.98,
  "duration": 3.5,
  "segments": [
    {"start": 0.0, "end": 3.5, "text": "Hello, this is a voice message."}
  ],
  "backend": "local",
  "success": true,
  "error": null
}
```

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_BACKEND` | `auto` | `auto`, `local`, or `openai` |
| `WHISPER_MODEL` | `base` | Whisper model size (see below) |
| `OPENAI_API_KEY` | -- | Required for `openai` backend |
| `TELEGRAM_BOT_TOKEN` | -- | Required for `transcribe_telegram_voice` |
| `WHISPER_LANGUAGE` | auto-detect | ISO-639-1 language code |

## How It Works

```
                         MCP Client (Claude)
                              |
                         [MCP stdio]
                              |
                    whisper-telegram-mcp
                    /         |         \
                   /          |          \
      transcribe_audio  transcribe_     speak_text
                        telegram_voice      |
              |               |        [macOS say]
              |         [Bot API DL]        |
              +--------+------+        [PyAV encode]
                       |                    |
                 auto_transcribe()        .ogg file
                  /           \
           LocalBackend    OpenAIBackend
           (faster-whisper)  (Whisper API)
```

1. Claude sends a tool call via MCP (stdio transport)
2. For Telegram voice messages, the file is downloaded via Bot API
3. `auto_transcribe()` picks the best available backend
4. Transcription result is returned as structured JSON

## Local vs OpenAI

| | Local (faster-whisper) | OpenAI API |
|---|---|---|
| **Cost** | Free | $0.006/min |
| **Privacy** | All data stays on device | Audio sent to OpenAI |
| **Speed** | ~1-10s depending on model | ~1-3s |
| **Setup** | Automatic (downloads model on first use) | Requires `OPENAI_API_KEY` |
| **Accuracy** | Excellent with `base` or larger | Excellent |
| **Offline** | Yes | No |

### Model Sizes

| Model | Parameters | Speed | Accuracy | VRAM |
|-------|-----------|-------|----------|------|
| `tiny` | 39M | Fastest | Lowest | ~1GB |
| `base` | 74M | Fast | Good | ~1GB |
| `small` | 244M | Moderate | Better | ~2GB |
| `medium` | 769M | Slow | High | ~5GB |
| `large-v3` | 1550M | Slowest | Highest | ~10GB |
| `turbo` | ~800M | Fast | High | ~6GB |

English-only variants (`tiny.en`, `base.en`, `small.en`, `medium.en`) are slightly more accurate for English.

## Development

```bash
git clone https://github.com/abid-mahdi/whisper-telegram-mcp.git
cd whisper-telegram-mcp
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run unit tests
pytest tests/ -v -m "not integration"

# Run integration tests (downloads ~150MB model on first run)
pytest tests/ -m integration -v

# Run with coverage
pytest tests/ --cov=src/whisper_telegram_mcp --cov-report=term-missing
```

### MCP Inspector

```bash
uvx mcp dev src/whisper_telegram_mcp/server.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Run tests (`pytest tests/ -v -m "not integration"`)
4. Commit with conventional commits (`feat:`, `fix:`, `docs:`, etc.)
5. Open a pull request

## License

[MIT](LICENSE)
