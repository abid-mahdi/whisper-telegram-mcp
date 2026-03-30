"""Telegram Bot API helpers for downloading voice messages."""
from __future__ import annotations

import os
import tempfile
from typing import Optional

import httpx

TELEGRAM_API = "https://api.telegram.org"
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB — Telegram voice message limit


class TelegramDownloadError(Exception):
    """Raised when a Telegram file download fails."""


async def get_file_path(bot_token: str, file_id: str) -> str:
    """Resolve a Telegram file_id to a downloadable file_path."""
    url = f"{TELEGRAM_API}/bot{bot_token}/getFile"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params={"file_id": file_id})
    if response.status_code != 200:
        raise TelegramDownloadError(f"HTTP {response.status_code} from Telegram API")
    data = response.json()
    if not data.get("ok"):
        raise TelegramDownloadError(f"Telegram API error: {data.get('description', 'Unknown error')}")
    return data["result"]["file_path"]


async def download_voice_message(
    bot_token: str,
    file_id: str,
    output_dir: Optional[str] = None,
) -> str:
    """Download a Telegram voice message to a local file.

    Returns the absolute path to the downloaded file.
    Caller is responsible for deleting the file when done.
    """
    file_path = await get_file_path(bot_token, file_id)
    download_url = f"{TELEGRAM_API}/file/bot{bot_token}/{file_path}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(download_url)

    if response.status_code != 200:
        raise TelegramDownloadError(f"Failed to download file: HTTP {response.status_code}")

    if len(response.content) > MAX_FILE_SIZE_BYTES:
        raise TelegramDownloadError(
            f"File too large ({len(response.content) / 1024 / 1024:.1f}MB). Max {MAX_FILE_SIZE_BYTES // 1024 // 1024}MB."
        )

    ext = os.path.splitext(file_path)[1] or ".oga"
    save_dir = output_dir or tempfile.gettempdir()
    os.makedirs(save_dir, exist_ok=True)

    fd, local_path = tempfile.mkstemp(suffix=ext, dir=save_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(response.content)
    except Exception:
        os.unlink(local_path)
        raise

    return local_path
