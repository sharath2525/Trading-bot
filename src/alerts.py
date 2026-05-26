"""Push alerting module — Telegram only. Fails silently if not configured.

Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable.
If either is unset, all send_alert() calls are no-ops.
"""

import asyncio
import logging
import os

import aiohttp
from dotenv import load_dotenv

load_dotenv()

_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
_ENABLED: bool = bool(_BOT_TOKEN and _CHAT_ID)

if not _ENABLED:
    logging.info("[ALERTS] Telegram not configured — push alerts disabled (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID to enable)")


async def send_alert(message: str) -> None:
    """Send a Telegram message. Silent no-op if not configured."""
    if not _ENABLED:
        return
    url = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": _CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logging.warning("[ALERTS] Telegram send failed: %d — %s", resp.status, body[:200])
    except Exception as _e:
        logging.warning("[ALERTS] Telegram send error: %s", _e)


def send_alert_sync(message: str) -> None:
    """Synchronous wrapper for use in signal handlers or non-async contexts.

    BUG-v8-S4 FIX: replaced asyncio.ensure_future() which silently drops alerts when
    the event loop is shutting down (e.g., KILLSWITCH). Now uses a direct synchronous
    urllib HTTP call that does not depend on the event loop being alive. This ensures
    KILLSWITCH confirmation and shutdown alerts are always delivered.
    """
    if not _ENABLED:
        return
    import json as _json
    import urllib.request as _urlreq
    import urllib.error as _urlerr
    url = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"
    payload = _json.dumps({"chat_id": _CHAT_ID, "text": message, "parse_mode": "HTML"}).encode()
    try:
        req = _urlreq.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with _urlreq.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                logging.warning("[ALERTS] Telegram sync send failed: %d", resp.status)
    except _urlerr.URLError as _e:
        logging.warning("[ALERTS] Telegram sync send error: %s", _e)
    except Exception as _e:
        logging.warning("[ALERTS] Telegram sync send error: %s", _e)
