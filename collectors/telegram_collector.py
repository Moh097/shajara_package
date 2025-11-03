# collectors/telegram_collector.py
# Telegram → SQLite (env-only; cron-friendly)

# --- ensure repo root in PYTHONPATH ---
import os, sys
HERE = os.path.dirname(__file__)
CANDIDATE = os.path.abspath(os.path.join(HERE, ".."))
if os.path.isdir(os.path.join(CANDIDATE, "utils")) and CANDIDATE not in sys.path:
    sys.path.insert(0, CANDIDATE)
else:
    REPO_ROOT = os.path.abspath(os.getcwd())
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
# ---

import asyncio, hashlib
from datetime import datetime, timezone
from telethon.sync import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.types import User
from utils.sqlite_client import upsert_posts, ensure_schema
from modules.config import DB_PATH

def _get_env(name, default=""):
    v = os.environ.get(name, default)
    return v.strip() if isinstance(v, str) else v

API_ID_STR = _get_env("TELEGRAM_API_ID", "")
API_HASH = _get_env("TELEGRAM_API_HASH", "")
STRING = _get_env("TELEGRAM_STRING_SESSION", "")

session_file = _get_env("TELEGRAM_SESSION_FILE", ".telegram_session")
if not STRING and os.path.exists(session_file):
    try:
        with open(session_file, "r", encoding="utf-8") as f:
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

CHANNELS = [c.strip() for c in _get_env("TG_CHANNELS", "dmski_1,akbardrwz,Druzeresistance").split(",") if c.strip()]
MAX_POSTS = int(_get_env("TG_MAX_POSTS", "200"))
PREFILTER = [k for k in _get_env("TG_PREFILTER", "السويداء,الساحل,اللاذقية,طرطوس,قتل,اشتباك,طائفي").split(",") if k]

def build_urls(message):
    source_url = ""
    post_url = ""
    chat = getattr(message, "chat", None)
    if chat and getattr(chat, "username", None):
        uname = chat.username
        source_url = f"https://t.me/{uname}"
        post_url = f"https://t.me/{uname}/{message.id}"
    else:
        chan_id = None
        if hasattr(message, "peer_id") and hasattr(message.peer_id, "channel_id"):
            chan_id = message.peer_id.channel_id
        elif chat and hasattr(chat, "id"):
            try: chan_id = abs(int(chat.id))
            except Exception: chan_id = None
        if chan_id:
            source_url = f"https://t.me/c/{chan_id}/{message.id}"
            post_url  = f"https://t.me/c/{chan_id}/{message.id}"
    return source_url, post_url

def extract_row(message):
    msg_date = ""
    if message.date:
        try: msg_date = message.date.astimezone(timezone.utc).isoformat()
        except Exception: msg_date = message.date.isoformat()

    source_name = ""
    source_url, post_url = build_urls(message)
    chat = getattr(message, "chat", None)
    if chat and getattr(chat, "title", None):
        source_name = chat.title

    author = ""
    if message.sender and isinstance(message.sender, User):
        if message.sender.username:
            author = message.sender.username
        elif message.sender.first_name and message.sender.last_name:
            author = f"{message.sender.first_name} {message.sender.last_name}"
        elif message.sender.first_name:
            author = message.sender.first_name

    text = message.text or ""
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
        "media_urls": "Media present (URL TBD)" if message.media else "",
        "shares": None,
        "likes": None,
        "comments": None,
        "collected_at_utc": datetime.utcnow().isoformat(),
        "collector": "SHAJARA-Agent",
        "hash": hashlib.sha256((text or "").encode("utf-8")).hexdigest() if text else None,
        "notes": "",
    }

async def run():
    ensure_schema(DB_PATH)
    rows = []
    try:
        async with TelegramClient(StringSession(STRING), API_ID, API_HASH) as client:
            print("Telegram client started — fetching messages...")
            for ch in CHANNELS:
                if not ch: continue
                print(f"Scanning channel: {ch}")
                try:
                    async for message in client.iter_messages(ch):
                        if len(rows) >= MAX_POSTS: break
                        if not getattr(message, "text", None): continue
                        if PREFILTER:
                            t = message.text or ""
                            if not any(k in t for k in PREFILTER):
                                if message.id % 10 != 0:
                                    continue
                        rows.append(extract_row(message))
                    if len(rows) >= MAX_POSTS: break
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

if __name__ == "__main__":
    asyncio.run(run())
