from __future__ import annotations
import os, sqlite3
from typing import Optional, Tuple, List
import pandas as pd

from .config import DB_PATH, S
from utils.sqlite_client import connect

def _where_clause(
    since_utc: Optional[str],
    until_utc: Optional[str],
    search: Optional[str],
) -> Tuple[str, List]:
    clauses = ["1=1"]
    params: List = []
    if since_utc:
        clauses.append("datetime_utc >= ?")
        params.append(since_utc)
    if until_utc:
        clauses.append("datetime_utc <= ?")
        params.append(until_utc)
    if search:
        clauses.append("INSTR(COALESCE(text,''), ?) > 0")
        params.append(search)
    return " AND ".join(clauses), params

def count_telegram_posts(
    since_utc: Optional[str] = None,
    until_utc: Optional[str] = None,
    search: Optional[str] = None,
) -> int:
    where_sql, params = _where_clause(since_utc, until_utc, search)
    sql = f"SELECT COUNT(1) FROM posts WHERE source_name='telegram' AND {where_sql};"
    with connect() as con:
        row = con.execute(sql, params).fetchone()
        return int(row[0] if row else 0)

def fetch_telegram_posts(
    limit: int = 100,
    since_utc: Optional[str] = None,
    until_utc: Optional[str] = None,
    search: Optional[str] = None,
) -> pd.DataFrame:
    where_sql, params = _where_clause(since_utc, until_utc, search)
    sql = f"""
        SELECT
            id, datetime_utc, source_name, author, text, likes, shares, comments
        FROM posts
        WHERE source_name='telegram' AND {where_sql}
        ORDER BY datetime_utc DESC
        LIMIT ?
    """
    with connect() as con:
        return pd.read_sql_query(sql, con, params=params + [int(limit)])
