# collectors/telegram_collector.py
# Telegram → SQLite (env-only; cron-friendly)

from __future__ import annotations
import os, sys, asyncio, hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List

# Make package imports work when called as a script
HERE = os.path.dirname(__file__)
PKG_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.types import User

from utils.sqlite_client import upsert_posts, ensure_schema
from modules.config import DB_PATH

# ---------------- Time helpers (aware UTC, RFC3339) ----------------

def utc_now_iso() -> str:
    # ISO-8601 with 'Z', seconds precision
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

def to_utc_iso(dt: datetime) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (
        dt.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

# ---------------- Env helpers ----------------

def _get_env(name: str, default: str = "") -> str:
    v = os.environ.get(name, default)
    if isinstance(v, str):
        # strip accidental quotes/spaces in .env
        return v.strip().strip('"').strip("'")
    return v

API_ID_STR = _get_env("TELEGRAM_API_ID", "")
API_HASH   = _get_env("TELEGRAM_API_HASH", "")
STRING     = _get_env("TELEGRAM_STRING_SESSION", "")

# Optional local file fallback for the string session
SESSION_FILE = _get_env("TELEGRAM_SESSION_FILE", ".telegram_session")
if not STRING and os.path.exists(SESSION_FILE):
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            if raw:
                STRING = raw
                print("Info: Loaded TELEGRAM_STRING_SESSION from local file.")
    except Exception as e:
        print("Warning: couldn't read local session file:", e)

if not API_ID_STR or not API_HASH:
    print("ERROR: TELEGRAM_API_ID or TELEGRAM_API_HASH is missing.")
    raise SystemExit(2)
try:
    API_ID = int(API_ID_STR)
except Exception:
    print("ERROR: TELEGRAM_API_ID must be an integer.")
    raise SystemExit(2)
if not STRING:
    print("ERROR: TELEGRAM_STRING_SESSION is missing.")
    raise SystemExit(2)

print(f"DEBUG: TELEGRAM_STRING_SESSION length = {len(STRING)} (masked).")

# Channels / limits / prefilter
CHANNELS   = [c.strip() for c in _get_env("TG_CHANNELS", "dmski_1,akbardrwz,Druzeresistance").split(",") if c.strip()]
MAX_POSTS  = int(_get_env("TG_MAX_POSTS", "200"))
PREFILTER  = [k for k in _get_env("TG_PREFILTER", "السويداء,الساحل,اللاذقية,طرطوس,قتل,اشتباك,طائفي").split(",") if k]

# ---------------- Message helpers ----------------

def build_urls(message) -> tuple[str, str]:
    """
    Best-effort to produce public t.me links when possible.
    Falls back to /c/<id>/<msg> format for private/supergroup cases.
    """
    source_url = ""
    post_url   = ""

    chat = getattr(message, "chat", None)
    uname = getattr(chat, "username", None) if chat else None
    if uname:  # public channel/group
        source_url = f"https://t.me/{uname}"
        post_url   = f"https://t.me/{uname}/{message.id}"
        return source_url, post_url

    # private/supergroup: try to derive numeric id
    chan_id = None
    if hasattr(message, "peer_id") and hasattr(message.peer_id, "channel_id"):
        chan_id = message.peer_id.channel_id
    elif chat and hasattr(chat, "id"):
        try:
            chan_id = abs(int(chat.id))
        except Exception:
            chan_id = None

    if chan_id:
        source_url = f"https://t.me/c/{chan_id}"
        post_url   = f"https://t.me/c/{chan_id}/{message.id}"

    return source_url, post_url

def extract_row(message) -> Dict[str, Any]:
    # Timestamp
    msg_date = to_utc_iso(getattr(message, "date", None))

    # Source/channel title
    source_name = ""
    chat = getattr(message, "chat", None)
    if chat and getattr(chat, "title", None):
        source_name = chat.title

    # Author (if available)
    author = ""
    try:
        if message.sender and isinstance(message.sender, User):
            if message.sender.username:
                author = message.sender.username
            elif message.sender.first_name and message.sender.last_name:
                author = f"{message.sender.first_name} {message.sender.last_name}"
            elif message.sender.first_name:
                author = message.sender.first_name
    except Exception:
        pass

    # Content
    text = getattr(message, "text", None) or getattr(message, "message", "") or ""

    # URLs
    source_url, post_url = build_urls(message)

    # Row (keep the same schema you already use)
    return {
        "platform": "Telegram",
        "platform_post_id": str(message.id),
        "source_name": source_name,
        "source_url": source_url,
        "post_id": str(message.id),
        "post_url": post_url,
        "author": author,
        "text": text,
        "language": "ar",
        "datetime_utc": msg_date,
        "datetime_local": "",
        "admin_area": "",
        "locality": "",
        "geofenced_area": "",
        "tension_level": "",
        "media_urls": "Media present (URL TBD)" if getattr(message, "media", None) else "",
        "shares": None,
        "likes": None,
        "comments": None,
        "collected_at_utc": utc_now_iso(),
        "collector": "SHAJARA-Agent",
        "hash": hashlib.sha256((text or "").encode("utf-8")).hexdigest() if text else None,
        "notes": "",
    }

# ---------------- Main async run ----------------

async def run() -> None:
    ensure_schema(DB_PATH)
    rows: List[Dict[str, Any]] = []

    try:
        async with TelegramClient(StringSession(STRING), API_ID, API_HASH) as client:
            print("Telegram client started — fetching messages...")
            for ch in CHANNELS:
                if not ch:
                    continue
                print(f"Scanning channel: {ch}")
                try:
                    async for message in client.iter_messages(ch):
                        if len(rows) >= MAX_POSTS:
                            break
                        # only text posts (or with text + media)
                        t = getattr(message, "text", None) or getattr(message, "message", None)
                        if not t:
                            continue

                        # Prefilter by simple keyword list (fast throttle)
                        if PREFILTER:
                            if not any(k in t for k in PREFILTER):
                                # Sample 10% of non-matching posts to keep some variety
                                if (message.id % 10) != 0:
                                    continue

                        rows.append(extract_row(message))

                    if len(rows) >= MAX_POSTS:
                        break
                except Exception as e:
                    print(f"Warning: failed scanning {ch}: {e}")
    except Exception as e:
        print(f"ERROR: Telegram client failed to start or authenticate: {e}")
        return

    if rows:
        try:
            n = upsert_posts(rows, db_path=DB_PATH)
            print(f"Upserted {n} Telegram rows into SQLite: {DB_PATH}")
        except Exception as e:
            print(f"ERROR: SQLite upsert failed: {e}")
    else:
        print("No Telegram rows collected.")

# ---------------- Entrypoint ----------------

if __name__ == "__main__":
    asyncio.run(run())
