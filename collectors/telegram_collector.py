# collectors/telegram_collector.py
from __future__ import annotations
import os, sys, json, asyncio, logging
from pathlib import Path
from datetime import timezone
from typing import List, Tuple

# ---- Make sure sibling packages (modules/, utils/) are importable when run directly
ROOT = Path(__file__).resolve().parents[1]  # .../shajara_package
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.custom import Message  # type: ignore

from modules.config import export_secrets_to_env, S
from utils.sqlite_client import ensure_schema, upsert_posts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tg_collector")

def _parse_channels(raw: str | None) -> List[str]:
    if not raw:
        return []
    raw = raw.strip()
    if not raw:
        return []
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
    return parts

def _msg_metrics(m: Message) -> Tuple[int, int, int]:
    likes = 0
    shares = 0
    comments = 0
    try:
        if getattr(m, "reactions", None) and getattr(m.reactions, "results", None):
            likes = sum(int(r.count or 0) for r in m.reactions.results)
    except Exception:
        pass
    try:
        if getattr(m, "forwards", None) is not None:
            shares = int(m.forwards or 0)
    except Exception:
        pass
    try:
        if getattr(m, "replies", None) and getattr(m.replies, "replies", None) is not None:
            comments = int(m.replies.replies or 0)
    except Exception:
        pass
    return likes, shares, comments

async def _collect_for_channel(client: TelegramClient, channel: str, limit: int) -> List[tuple]:
    ent = await client.get_entity(channel)
    rows: List[tuple] = []
    cnt = 0
    async for m in client.iter_messages(ent, limit=limit):
        if not isinstance(m, Message):
            continue
        text = (m.message or "")
        dt = m.date.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if m.date else ""
        author = ""
        try:
            author = (getattr(m, "post_author", None)
                      or (m.sender.username if getattr(m, "sender", None) and getattr(m.sender, "username", None) else "")
                      or (m.sender.first_name if getattr(m, "sender", None) and getattr(m.sender, "first_name", None) else "")
                      or "")
        except Exception:
            pass
        likes, shares, comments = _msg_metrics(m)
        uid = f"tg:{channel}:{m.id}"
        rows.append((uid, dt, "telegram", author, text, likes, shares, comments))
        cnt += 1
    log.info("Channel %s -> fetched %d messages", channel, cnt)
    return rows

async def main():
    export_secrets_to_env()
    ensure_schema(None)

    api_id = S("TELEGRAM_API_ID")
    api_hash = S("TELEGRAM_API_HASH")
    session = S("TELEGRAM_STRING_SESSION")
    channels_raw = os.environ.get("TELEGRAM_CHANNELS") or S("TELEGRAM_CHANNELS", "")
    max_fetch = int(S("TELEGRAM_MAX_FETCH", "1000"))

    if not api_id or not api_hash or not session:
        raise RuntimeError("Missing TELEGRAM_API_ID / TELEGRAM_API_HASH / TELEGRAM_STRING_SESSION in secrets.toml")
    try:
        api_id = int(api_id)
    except Exception:
        raise RuntimeError("TELEGRAM_API_ID must be an integer")

    channels = _parse_channels(channels_raw)
    if not channels:
        raise RuntimeError("No channels specified. Set TELEGRAM_CHANNELS (env or .streamlit/secrets.toml)")

    log.info("Connecting to Telegram...")
    async with TelegramClient(StringSession(session), api_id, api_hash) as client:
        client.parse_mode = "html"
        all_rows: List[tuple] = []
        for ch in channels:
            try:
                rows = await _collect_for_channel(client, ch, limit=max_fetch)
                all_rows.extend(rows)
            except Exception as e:
                log.error("Failed channel %s: %s", ch, e)

    if not all_rows:
        log.warning("No messages collected.")
        return

    inserted = upsert_posts(all_rows)
    log.info("Upserted %d rows into SQLite (%s)", inserted, os.environ.get("SHAJARA_DB_PATH"))

if __name__ == "__main__":
    asyncio.run(main())
