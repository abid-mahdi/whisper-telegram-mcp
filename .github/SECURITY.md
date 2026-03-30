# Security Policy

## Reporting Security Issues

**Do not open public GitHub issues for security vulnerabilities.**

Email: [abid@abidmahdi.com](mailto:abid@abidmahdi.com)

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

You will receive a response within 48 hours.

## Security Practices

- API keys and bot tokens are loaded from environment variables — never hardcoded
- Temporary audio files are deleted immediately after transcription
- No telemetry, tracking, or analytics
- All data stays local when using the faster-whisper backend
- Logs go to stderr only — no credentials or audio content logged
