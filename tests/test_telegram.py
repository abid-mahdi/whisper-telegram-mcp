"""Tests for Telegram download helpers."""
import pytest
import httpx
import respx
from whisper_telegram_mcp.telegram import TelegramDownloadError, download_voice_message, get_file_path

BOT_TOKEN = "123456789:ABCDefGhIJklmNOPQrst"
FILE_ID = "AwACAgIAAxkBAAIBpg"
FILE_PATH = "voice/file_1.oga"


@pytest.mark.asyncio
async def test_download_success(tmp_path):
    """Successfully downloads and saves a voice message."""
    with respx.mock:
        respx.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getFile").mock(
            return_value=httpx.Response(200, json={
                "ok": True,
                "result": {"file_path": FILE_PATH, "file_id": FILE_ID, "file_size": 1024}
            })
        )
        respx.get(f"https://api.telegram.org/file/bot{BOT_TOKEN}/{FILE_PATH}").mock(
            return_value=httpx.Response(200, content=b"OGG_AUDIO_DATA")
        )
        output_path = await download_voice_message(
            bot_token=BOT_TOKEN, file_id=FILE_ID, output_dir=str(tmp_path)
        )
    assert output_path.endswith(".oga") or output_path.endswith(".ogg")
    with open(output_path, "rb") as f:
        assert f.read() == b"OGG_AUDIO_DATA"


@pytest.mark.asyncio
async def test_download_invalid_file_id():
    """Raises TelegramDownloadError on invalid file_id."""
    with respx.mock:
        respx.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getFile").mock(
            return_value=httpx.Response(200, json={"ok": False, "description": "Bad Request: file not found"})
        )
        with pytest.raises(TelegramDownloadError, match="file not found"):
            await download_voice_message(BOT_TOKEN, "INVALID_FILE_ID")


@pytest.mark.asyncio
async def test_download_http_error():
    """Raises TelegramDownloadError on HTTP 401."""
    with respx.mock:
        respx.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getFile").mock(
            return_value=httpx.Response(401, json={"ok": False, "description": "Unauthorized"})
        )
        with pytest.raises(TelegramDownloadError):
            await download_voice_message(BOT_TOKEN, FILE_ID)


@pytest.mark.asyncio
async def test_get_file_path_success():
    """get_file_path returns the file_path string."""
    with respx.mock:
        respx.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getFile").mock(
            return_value=httpx.Response(200, json={"ok": True, "result": {"file_path": FILE_PATH}})
        )
        path = await get_file_path(BOT_TOKEN, FILE_ID)
    assert path == FILE_PATH
