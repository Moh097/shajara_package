# modules/data_access.py

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.supabase_client import select

# Table name used by both app + collectors
POSTS_TABLE = os.environ.get("SHAJARA_TABLE_NAME", "posts")

# Columns app cares about
POSTS_SELECT_COLUMNS = (
    "id,datetime_utc,source_name,author,text,likes,shares,comments"
)


def _build_datetime_filter(
    since_utc: Optional[str],
    until_utc: Optional[str],
) -> Optional[List[str]]:
    """
    Build PostgREST-style filter for datetime_utc:
      datetime_utc=gte.XXXXX
      datetime_utc=lte.YYYYY

    If both are provided we return a list so requests encodes them as:
      &datetime_utc=gte...&datetime_utc=lte...
    """
    filters: List[str] = []
    if since_utc:
        filters.append(f"gte.{since_utc}")
    if until_utc:
        filters.append(f"lte.{until_utc}")
    return filters or None


def fetch_telegram_posts(
    limit: int = 100,
    *,
    since_utc: Optional[str] = None,
    until_utc: Optional[str] = None,
    search: Optional[str] = None,
) -> pd.DataFrame:
    """
    Main loader used by app.py to get posts.

    Despite the name, this returns posts from `posts` table regardless
    of platform (telegram/facebook). If you want to filter only Telegram
    later, add `&platform=eq.telegram` here.
    """
    params: Dict[str, Any] = {
        "select": POSTS_SELECT_COLUMNS,
        "order": "datetime_utc.desc",
        "limit": str(max(1, int(limit))),
    }

    dt_filter = _build_datetime_filter(since_utc, until_utc)
    if dt_filter is not None:
        params["datetime_utc"] = dt_filter

    if search:
        # Case-insensitive "contains" on text column
        params["text"] = f"ilike.%{search}%"

    rows, _ = select(POSTS_TABLE, params, count=False)
    if not rows:
        return pd.DataFrame(
            columns=["id", "datetime_utc", "source_name",
                     "author", "text", "likes", "shares", "comments"]
        )
    return pd.DataFrame(rows)


def db_total_rows() -> int:
    """
    Exact row count from Supabase using Content-Range / count=exact.
    """
    params: Dict[str, Any] = {
        "select": "id",
        "limit": "1",
    }
    _, total = select(POSTS_TABLE, params, count=True)
    return int(total or 0)


def fetch_latest_for_channels(
    channels: List[str],
    limit_rows: int = 200,
) -> pd.DataFrame:
    """
    Used by the 'collection summary & preview' block in app.py.
    Reads latest posts for a given list of channels from Supabase.
    """
    channels = [c.strip() for c in channels if c and c.strip()]
    if not channels:
        return pd.DataFrame(
            columns=["id", "datetime_utc", "source_name",
                     "author", "text", "likes", "shares", "comments"]
        )

    params: Dict[str, Any] = {
        "select": POSTS_SELECT_COLUMNS,
        "order": "datetime_utc.desc",
        "limit": str(max(1, int(limit_rows))),
        # PostgREST IN syntax, e.g. in.(foo,bar)
        "source_name": "in.(" + ",".join(channels) + ")",
    }

    rows, _ = select(POSTS_TABLE, params, count=False)
    if not rows:
        return pd.DataFrame(
            columns=["id", "datetime_utc", "source_name",
                     "author", "text", "likes", "shares", "comments"]
        )
    return pd.DataFrame(rows)
