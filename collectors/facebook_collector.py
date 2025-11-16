# collectors/facebook_collector.py

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List
from pathlib import Path

try:
    from facebook_scraper import get_posts
except ImportError:
    get_posts = None  # we’ll handle this gracefully

from modules.config import S
from utils.supabase_client import upsert

log = logging.getLogger("facebook_collector")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)

PKG_ROOT = Path(__file__).resolve().parents[1]
POSTS_TABLE = os.environ.get("SHAJARA_TABLE_NAME", S("SHAJARA_TABLE_NAME", "posts")) or "posts"


def _parse_list(raw: str) -> List[str]:
    raw = raw.strip()
    if not raw:
        return []
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    parts = [p.strip() for p in raw.replace("\r", "").replace("\n", ",").split(",")]
    return [p for p in parts if p]


def _load_env_list(key: str) -> List[str]:
    # 1) explicit env wins
    raw_env = os.environ.get(key)
    if raw_env is not None and str(raw_env).strip():
        return _parse_list(str(raw_env))

    # 2) .streamlit/channels.json for TELEGRAM_CHANNELS (reused as Facebook pages)
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


def run_facebook_collector() -> None:
    if get_posts is None:
        print("facebook_scraper not installed; skipping Facebook collector.")
        print("Upserted 0 rows")
        return

    # Reuse TELEGRAM_* env as generic "sources" for FB too
    pages = _load_env_list("TELEGRAM_CHANNELS")
    search_terms = _load_env_list("TELEGRAM_SEARCH_TERMS")

    limit_raw = os.environ.get("TELEGRAM_LIMIT")
    if not (limit_raw and str(limit_raw).strip()):
        limit_raw = S("TELEGRAM_LIMIT", S("TELEGRAM_MAX_FETCH", "1000"))
    try:
        limit = int(str(limit_raw))
    except Exception:
        limit = 1000

    if not pages:
        print("No Facebook pages configured (using TELEGRAM_CHANNELS). Nothing to do.")
        print("Upserted 0 rows")
        return

    print(f"Using Facebook pages: {pages}")
    print(f"Search terms: {search_terms or '∅'}")
    print(f"Per-page limit: {limit}")

    all_rows: List[Dict[str, Any]] = []

    for page in pages:
        page = page.strip()
        if not page:
            continue

        fetched = 0
        rows_for_page: List[Dict[str, Any]] = []

        try:
            for post in get_posts(page=page, pages=1, extra_info=True):
                if fetched >= limit:
                    break

                text = (post.get("text") or "") + " " + (post.get("post_text") or "")
                if not _match_search_terms(text, search_terms):
                    continue

                dt = post.get("time")
                if dt is not None:
                    dt_iso = dt.isoformat()
                else:
                    dt_iso = None

                row = {
                    "platform": "facebook",
                    "source_name": page,
                    "source_id": str(post.get("post_id") or ""),
                    "datetime_utc": dt_iso,
                    "author": post.get("username") or post.get("user_id") or None,
                    "text": text,
                    "likes": int(post.get("likes") or 0),
                    "shares": int(post.get("shares") or 0),
                    "comments": int(post.get("comments") or 0),
                    "raw": {
                        "post_url": post.get("post_url"),
                    },
                }

                rows_for_page.append(row)
                fetched += 1

        except Exception as e:
            log.error("Failed scraping Facebook page %s: %s", page, e)
            continue

        print(f"Channel {page} -> fetched {fetched} messages")
        all_rows.extend(rows_for_page)

    if not all_rows:
        print("Upserted 0 rows (no Facebook posts matched filters)")
        return

    inserted = upsert(POSTS_TABLE, all_rows)
    print(f"Upserted {inserted} rows")


def main():
    run_facebook_collector()


if __name__ == "__main__":
    main()
