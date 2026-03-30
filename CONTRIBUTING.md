# Contributing

Contributions are welcome. Here's how to get started.

## Workflow

1. Fork the repo and create a feature branch from `main`
2. Make your changes
3. Run the tests: `pytest`
4. Open a pull request with a clear description of what you changed and why

## Development Setup

```bash
git clone https://github.com/your-fork/whisper-telegram-mcp
cd whisper-telegram-mcp
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in your credentials.

## Tests

```bash
pytest                    # all unit tests
pytest -m "not integration"  # skip tests that download Whisper models
pytest --cov              # with coverage report
```

## Commit Style

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` no behaviour change
- `test:` tests only
- `chore:` build/config changes

## What to Contribute

- Bug fixes and reliability improvements are always welcome
- New TTS or transcription backends
- Improved test coverage
- Documentation improvements

For larger changes, open an issue first to discuss the approach.
