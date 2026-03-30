# whisper-telegram-mcp

> Transcribe Telegram voice messages with Whisper -- as an MCP tool for Claude

[![CI](https://github.com/abid-mahdi/whisper-telegram-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/abid-mahdi/whisper-telegram-mcp/actions)
[![Python Version](https://img.shields.io/pypi/pyversions/whisper-telegram-mcp)](https://pypi.org/project/whisper-telegram-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-compatible-brightgreen)](https://modelcontextprotocol.io)

An [MCP](https://modelcontextprotocol.io) server that transcribes audio files and Telegram voice messages using OpenAI's Whisper speech recognition. Works with Claude Desktop, Claude Code, and any MCP-compatible client.

## What It Does

- **Transcribe local audio files** -- OGG, WAV, MP3, FLAC, and more
- **Transcribe Telegram voice messages** -- pass a `file_id`, get text back
- **Two backends** -- local inference with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (free, private) or OpenAI Whisper API (cloud)
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

### Response Format

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
                         /         \
                        /           \
              transcribe_audio   transcribe_telegram_voice
                      |                    |
                      |            [Download via Bot API]
                      |                    |
                      +--------+-----------+
                               |
                         auto_transcribe()
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
