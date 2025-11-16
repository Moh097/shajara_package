# collectors/telegram_collector.py

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List
from datetime import timezone
from pathlib import Path

from telethon.sync import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.custom.message import Message

from modules.config import S
from utils.supabase_client import upsert

log = logging.getLogger("telegram_collector")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)

PKG_ROOT = Path(__file__).resolve().parents[1]
POSTS_TABLE = os.environ.get("SHAJARA_TABLE_NAME", S("SHAJARA_TABLE_NAME", "posts")) or "posts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_list(raw: str) -> List[str]:
    raw = raw.strip()
    if not raw:
        return []
    # Try JSON list first
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    # Fallback: newline/comma separated
    parts = [p.strip() for p in raw.replace("\r", "").replace("\n", ",").split(",")]
    return [p for p in parts if p]


def _load_env_list(key: str) -> List[str]:
    """
    Order:
      1) Explicit env var
      2) .streamlit/channels.json (for TELEGRAM_CHANNELS)
      3) .streamlit/secrets.toml via S()
    """
    # 1) explicit env wins
    raw_env = os.environ.get(key)
    if raw_env is not None and str(raw_env).strip():
        return _parse_list(str(raw_env))

    # 2) .streamlit/channels.json for TELEGRAM_CHANNELS
    if key == "TELEGRAM_CHANNELS":
        ch_file = PKG_ROOT / ".streamlit" / "channels.json"
        if ch_file.exists():
            try:
                arr = json.loads(ch_file.read_text(encoding="utf-8"))
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if str(x).strip()]
            except Exception:
                pass

    # 3) secrets.toml via S()
    raw_secret = S(key, "") or ""
    if raw_secret:
        return _parse_list(str(raw_secret))

    return []


def _get_telegram_client() -> TelegramClient:
    api_id_str = (os.environ.get("TELEGRAM_API_ID") or S("TELEGRAM_API_ID", "") or "").strip()
    api_hash = (os.environ.get("TELEGRAM_API_HASH") or S("TELEGRAM_API_HASH", "") or "").strip()
    session_str = (os.environ.get("TELEGRAM_STRING_SESSION") or S("TELEGRAM_STRING_SESSION", "") or "").strip()

    missing: List[str] = []
    if not api_id_str:
        missing.append("TELEGRAM_API_ID")
    if not api_hash:
        missing.append("TELEGRAM_API_HASH")
    if not session_str:
        missing.append("TELEGRAM_STRING_SESSION")

    if missing:
        raise RuntimeError(
            "Missing Telegram configuration: "
            + ", ".join(missing)
            + ". Set them in environment variables or .streamlit/secrets.toml."
        )

    api_id = int(api_id_str)
    return TelegramClient(StringSession(session_str), api_id, api_hash)


def _msg_to_row(channel_label: str, msg: Message) -> Dict[str, Any]:
    # datetime_utc as ISO string
    dt = msg.date
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_iso = dt.isoformat().replace("+00:00", "Z")

    text = msg.message or ""
    author = None
    try:
        if msg.sender and getattr(msg.sender, "username", None):
            author = msg.sender.username
        elif msg.sender and getattr(msg.sender, "first_name", None):
            author = msg.sender.first_name
    except Exception:
        author = None

    # Extract *numeric* metrics only – no Telethon objects in JSON
    views = getattr(msg, "views", None)
    forwards = getattr(msg, "forwards", None)
    replies_obj = getattr(msg, "replies", None)
    replies_count = getattr(replies_obj, "replies", None)

    try:
        likes = int(views or 0)
    except Exception:
        likes = 0
    try:
        shares = int(forwards or 0)
    except Exception:
        shares = 0
    try:
        comments = int(replies_count or 0)
    except Exception:
        comments = 0

    raw = {
        "views": views,
        "forwards": forwards,
        "replies": replies_count,
    }

    return {
        "platform": "telegram",
        "source_name": channel_label,
        "source_id": str(msg.id),
        "datetime_utc": dt_iso,
        "author": author,
        "text": text,
        "likes": likes,
        "shares": shares,
        "comments": comments,
        "raw": raw,
    }


def _match_search_terms(text: str, terms: List[str]) -> bool:
    if not terms:
        return True
    t = text or ""
    if not t:
        return False
    lower = t.lower()
    for term in terms:
        if not term:
            continue
        if term.lower() in lower:
            return True
    return False


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------

def run_telegram_collector() -> None:
    channels = _load_env_list("TELEGRAM_CHANNELS")
    search_terms = _load_env_list("TELEGRAM_SEARCH_TERMS")

    # limit: env TELEGRAM_LIMIT > secrets (TELEGRAM_LIMIT or TELEGRAM_MAX_FETCH) > default 1000
    limit_raw = os.environ.get("TELEGRAM_LIMIT")
    if not (limit_raw and str(limit_raw).strip()):
        limit_raw = S("TELEGRAM_LIMIT", S("TELEGRAM_MAX_FETCH", "1000"))
    try:
        limit = int(str(limit_raw))
    except Exception:
        limit = 1000

    if not channels:
        print("No channels configured (TELEGRAM_CHANNELS is empty). Nothing to do.")
        return

    print(f"Using channels: {channels}")
    print(f"Search terms: {search_terms or '∅'}")
    print(f"Per-channel limit: {limit}")

    client = _get_telegram_client()
    client.connect()

    all_rows: List[Dict[str, Any]] = []
    try:
        for ch in channels:
            ch = ch.strip()
            if not ch:
                continue

            try:
                entity = client.get_entity(ch)
            except Exception as e:
                log.error("Failed to get entity for %s: %s", ch, e)
                continue

            fetched = 0
            rows_for_channel: List[Dict[str, Any]] = []

            # Pull last `limit` messages and filter by search terms in Python
            for msg in client.iter_messages(entity, limit=limit):
                if not isinstance(msg, Message):
                    continue
                text = msg.message or ""
                if not _match_search_terms(text, search_terms):
                    continue
                row = _msg_to_row(ch, msg)
                rows_for_channel.append(row)
                fetched += 1

            print(f"Channel {ch} -> fetched {fetched} messages")
            all_rows.extend(rows_for_channel)
    finally:
        client.disconnect()

    if not all_rows:
        print("Upserted 0 rows (no messages matched filters)")
        return

    inserted = upsert(POSTS_TABLE, all_rows)
    print(f"Upserted {inserted} rows")


def main():
    run_telegram_collector()


if __name__ == "__main__":
    main()
