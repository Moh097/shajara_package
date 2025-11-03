# utils/sqlite_client.py
# Minimal SQLite helper for posts table with stable dedupe.

import os
import sqlite3
from typing import Iterable, Dict, Any, List

DEFAULT_DB = os.environ.get("SHAJARA_DB_PATH", "shajara.db")

POSTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  platform TEXT,
  platform_post_id TEXT,
  source_name TEXT,
  source_url TEXT,
  post_id TEXT,
  post_url TEXT,
  author TEXT,
  text TEXT,
  language TEXT,
  datetime_utc TEXT,
  datetime_local TEXT,
  admin_area TEXT,
  locality TEXT,
  geofenced_area TEXT,
  tension_level TEXT,
  media_urls TEXT,
  shares INTEGER,
  likes INTEGER,
  comments INTEGER,
  collected_at_utc TEXT,
  collector TEXT,
  hash TEXT,
  notes TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_posts_platform_pid
  ON posts(platform, platform_post_id);
CREATE INDEX IF NOT EXISTS idx_posts_datetime
  ON posts(datetime_utc);
CREATE INDEX IF NOT EXISTS idx_posts_tension
  ON posts(tension_level);
"""

UPSERT_SQL = """
INSERT INTO posts (
  platform, platform_post_id, source_name, source_url, post_id, post_url,
  author, text, language, datetime_utc, datetime_local, admin_area, locality,
  geofenced_area, tension_level, media_urls, shares, likes, comments,
  collected_at_utc, collector, hash, notes
) VALUES (
  :platform, :platform_post_id, :source_name, :source_url, :post_id, :post_url,
  :author, :text, :language, :datetime_utc, :datetime_local, :admin_area, :locality,
  :geofenced_area, :tension_level, :media_urls, :shares, :likes, :comments,
  :collected_at_utc, :collector, :hash, :notes
)
ON CONFLICT(platform, platform_post_id) DO UPDATE SET
  source_name=excluded.source_name,
  source_url=excluded.source_url,
  post_url=excluded.post_url,
  author=excluded.author,
  text=excluded.text,
  language=excluded.language,
  datetime_utc=excluded.datetime_utc,
  datetime_local=excluded.datetime_local,
  admin_area=excluded.admin_area,
  locality=excluded.locality,
  geofenced_area=excluded.geofenced_area,
  tension_level=excluded.tension_level,
  media_urls=excluded.media_urls,
  shares=excluded.shares,
  likes=excluded.likes,
  comments=excluded.comments,
  collected_at_utc=excluded.collected_at_utc,
  collector=excluded.collector,
  hash=excluded.hash,
  notes=excluded.notes;
"""

def ensure_schema(db_path: str = DEFAULT_DB) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.executescript(POSTS_SCHEMA)
        con.commit()
    finally:
        con.close()

def _to_int(v):
    if v is None: return None
    if isinstance(v, int): return v
    try:
        s = str(v)
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else None
    except Exception:
        return None

def upsert_posts(rows: Iterable[Dict[str, Any]], db_path: str = DEFAULT_DB, batch_size: int = 200) -> int:
    rows = list(rows or [])
    if not rows:
        return 0
    ensure_schema(db_path)
    clean_rows: List[Dict[str, Any]] = []
    for r in rows:
        c = dict(r)
        c["shares"] = _to_int(c.get("shares"))
        c["likes"] = _to_int(c.get("likes"))
        c["comments"] = _to_int(c.get("comments"))
        clean_rows.append(c)

    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        for i in range(0, len(clean_rows), batch_size):
            cur.executemany(UPSERT_SQL, clean_rows[i:i+batch_size])
        con.commit()
    finally:
        con.close()
    return len(rows)
