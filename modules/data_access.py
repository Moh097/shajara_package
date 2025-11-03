from __future__ import annotations
from typing import Optional
import sqlite3
import pandas as pd
from .config import DB_PATH
from utils.sqlite_client import ensure_schema

def fetch_telegram_posts(limit: int = 500,
                         since_utc: Optional[str] = None,
                         until_utc: Optional[str] = None,
                         search: Optional[str] = None) -> pd.DataFrame:
    """
    Read posts from local SQLite DB where platform='Telegram'.
    Ensures schema exists before querying.
    """
    # Make sure the table exists (fixes "no such table: posts")
    ensure_schema(DB_PATH)

    con = sqlite3.connect(DB_PATH)
    try:
        q = """
        SELECT
          id, platform, platform_post_id, source_name, source_url,
          post_id, post_url, author, text, language,
          datetime_utc, collected_at_utc, likes, shares, comments, hash
        FROM posts
        WHERE platform='Telegram'
        """
        params = []

        if since_utc:
            q += " AND datetime_utc >= ?"
            params.append(since_utc)
        if until_utc:
            q += " AND datetime_utc <= ?"
            params.append(until_utc)
        if search:
            q += " AND text LIKE ?"
            params.append(f"%{search}%")

        q += " ORDER BY datetime_utc DESC LIMIT ?"
        params.append(int(limit))

        df = pd.read_sql_query(q, con, params=params)
    finally:
        con.close()

    for col in ("datetime_utc", "collected_at_utc"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df
