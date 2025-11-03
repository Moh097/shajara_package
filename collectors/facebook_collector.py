# collectors/facebook_collector.py
# Facebook → SQLite (facebook_scraper; env-only)

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

import re, json, hashlib
from datetime import datetime, timezone
from typing import Iterable, Dict, Any, Tuple, Optional

from facebook_scraper import get_posts, set_user_agent
from utils.sqlite_client import upsert_posts, ensure_schema
from modules.config import DB_PATH

PAGE_URLS = [u.strip() for u in os.environ.get("FB_PAGES", "https://www.facebook.com/Suwayda24,https://www.facebook.com/groups/zero0nine9").split(",") if u.strip()]
POSTS_LIMIT = int(os.environ.get("FB_LIMIT", "200") or 200)
USER_AGENT = os.environ.get("FB_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
FB_COOKIES_JSON = os.environ.get("FB_COOKIES_JSON", "").strip()
set_user_agent(USER_AGENT)

def _parse_cookies(raw: str) -> Optional[dict]:
    if not raw: return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            jar = {}
            for c in obj:
                n, v = c.get("name"), c.get("value")
                if n and v is not None: jar[n] = v
            return jar or None
    except Exception:
        return None

COOKIES = _parse_cookies(FB_COOKIES_JSON)

def _identify(url: str):
    if not url: return "unknown", "", url
    from urllib.parse import urlparse, parse_qs
    url = re.sub(r"^(https?://)(m\.|mbasic\.)?facebook\.com", r"\1www.facebook.com", url.strip())
    p = urlparse(url)
    path = (p.path or "").strip("/")
    if not path:
        return "account", "", url
    if path.startswith("groups/"):
        parts = path.split("/")
        ident = parts[1] if len(parts) > 1 else ""
        return "group", ident, f"https://www.facebook.com/groups/{ident}"
    if "profile.php" in (p.path or ""):
        q = parse_qs(p.query or "")
        pid = q.get("id", [""])[0]
        return "account", pid, f"https://www.facebook.com/profile.php?id={pid}"
    m = re.search(r"/people/[^/]+/(\d+)", p.path or "")
    if m:
        pid = m.group(1)
        return "account", pid, f"https://www.facebook.com/people/x/{pid}"
    ident = path.split("/")[0]
    return "account", ident, f"https://www.facebook.com/{ident}"

def _row_from_post(post: Dict[str, Any], source_name: str, source_url: str) -> Dict[str, Any]:
    text = (post.get("text") or "").strip()
    t = post.get("time")
    if isinstance(t, datetime):
        dt_utc = t.astimezone(timezone.utc).isoformat()
    else:
        dt_utc = None

    imgs = post.get("images") or []
    if isinstance(imgs, str):
        media = imgs
    else:
        try: media = ",".join(imgs) if imgs else ""
        except Exception: media = ""

    return {
        "platform": "Facebook",
        "platform_post_id": str(post.get("post_id") or ""),
        "source_name": source_name or (post.get("username") or ""),
        "source_url": source_url,
        "post_id": str(post.get("post_id") or ""),
        "post_url": post.get("post_url") or "",
        "author": post.get("username") or "",
        "text": text,
        "language": "",
        "datetime_utc": dt_utc,
        "datetime_local": "",
        "admin_area": "",
        "locality": "",
        "geofenced_area": "",
        "tension_level": "",
        "media_urls": media,
        "shares": post.get("shares"),
        "likes": post.get("likes"),
        "comments": post.get("comments"),
        "collected_at_utc": datetime.utcnow().isoformat(),
        "collector": "SHAJARA-Agent",
        "hash": hashlib.sha256((text or "").encode("utf-8")).hexdigest() if text else None,
        "notes": "",
    }

def _iter_source_posts(url: str, limit: int):
    kind, ident, canonical = _identify(url)
    if not ident: return []
    opts = {"posts_per_page": 200}
    rows = 0
    gen = get_posts(group=ident, cookies=COOKIES, options=opts, pages=1000) if kind=="group" else get_posts(ident, cookies=COOKIES, options=opts, pages=1000)
    for post in gen:
        yield _row_from_post(post, source_name=(f"group:{ident}" if kind == "group" else ident), source_url=canonical)
        rows += 1
        if rows >= limit: break

def main():
    ensure_schema(DB_PATH)
    all_rows = []
    for u in PAGE_URLS:
        try:
            for row in _iter_source_posts(u, POSTS_LIMIT):
                all_rows.append(row)
                if len(all_rows) >= POSTS_LIMIT: break
        except Exception as e:
            print(f"Warning: failed scraping {u}: {e}")
        if len(all_rows) >= POSTS_LIMIT: break

    if all_rows:
        try:
            n = upsert_posts(all_rows, db_path=DB_PATH)
            print(f"Upserted {n} Facebook rows into SQLite: {DB_PATH}")
        except Exception as e:
            print(f"ERROR: SQLite upsert failed: {e}")
    else:
        print("No Facebook rows collected.")

if __name__ == "__main__":
    main()
