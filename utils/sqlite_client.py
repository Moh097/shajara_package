from __future__ import annotations
import os, sqlite3, logging
from pathlib import Path
from typing import Iterable, Sequence

from modules.config import DB_PATH, S

log = logging.getLogger("sqlite_client")
PKG_ROOT = Path(__file__).resolve().parents[1]  # .../shajara_package


def _absify(p: str) -> str:
    """Keep env as 'shajara.db' but always open relative to the package root."""
    p = (p or "").strip() or "shajara.db"
    return str((Path(p) if os.path.isabs(p) else (PKG_ROOT / p)).resolve())


def _db_path() -> str:
    # env > secrets > default, then anchor
    p = os.environ.get("SHAJARA_DB_PATH") or S("SHAJARA_DB_PATH", DB_PATH) or "shajara.db"
    return _absify(p)


def connect() -> sqlite3.Connection:
    p = _db_path()
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=30, check_same_thread=False)
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA foreign_keys=ON;")
    except Exception:
        pass
    return con


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;", (name,)
    ).fetchone()
    return bool(row)


def _posts_schema_ok(con: sqlite3.Connection) -> bool:
    if not _table_exists(con, "posts"):
        return False
    info = con.execute("PRAGMA table_info(posts);").fetchall()
    expected = [
        ("id", "TEXT"),
        ("datetime_utc", "TEXT"),
        ("source_name", "TEXT"),
        ("author", "TEXT"),
        ("text", "TEXT"),
        ("likes", "INTEGER"),
        ("shares", "INTEGER"),
        ("comments", "INTEGER"),
    ]
    if len(info) != len(expected):
        return False
    byname = {r[1].lower(): (r[1], (r[2] or "").upper()) for r in info}
    for name, etype in expected:
        if name not in byname:
            return False
        _, got_type = byname[name]
        if etype == "TEXT" and "CHAR" in got_type:
            continue
        if etype != got_type:
            return False
    return True


def _create_posts(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS posts(
            id TEXT PRIMARY KEY,
            datetime_utc TEXT NOT NULL,
            source_name TEXT NOT NULL,
            author TEXT,
            text TEXT,
            likes INTEGER,
            shares INTEGER,
            comments INTEGER
        );
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS ix_posts_datetime ON posts(datetime_utc DESC);")
    con.execute("CREATE INDEX IF NOT EXISTS ix_posts_source ON posts(source_name, datetime_utc DESC);")


def _migrate_posts_if_needed(con: sqlite3.Connection) -> None:
    if not _table_exists(con, "posts"):
        _create_posts(con)
        con.commit()
        log.info("SQLite: created posts table.")
        return

    if _posts_schema_ok(con):
        return

    log.warning("SQLite: migrating posts table to the new schema (TEXT id, 8 cols).")
    con.execute("BEGIN IMMEDIATE;")
    try:
        con.execute(
            """
            CREATE TABLE posts_new(
                id TEXT PRIMARY KEY,
                datetime_utc TEXT NOT NULL,
                source_name TEXT NOT NULL,
                author TEXT,
                text TEXT,
                likes INTEGER,
                shares INTEGER,
                comments INTEGER
            );
            """
        )

        cols = [r[1].lower() for r in con.execute("PRAGMA table_info(posts);").fetchall()]
        has = lambda c: c in cols

        sel_id          = "CAST(id AS TEXT)" if has("id") else "CAST(random() AS TEXT)"
        sel_datetime    = "datetime_utc" if has("datetime_utc") else "''"
        sel_source      = "source_name"  if has("source_name")  else "'telegram'"
        sel_author      = "author"       if has("author")       else "''"
        sel_text        = "text"         if has("text")         else "''"
        sel_likes       = "CAST(likes AS INTEGER)"   if has("likes")   else "0"
        sel_shares      = "CAST(shares AS INTEGER)"  if has("shares")  else "0"
        sel_comments    = "CAST(comments AS INTEGER)"if has("comments")else "0"

        con.execute(
            f"""
            INSERT OR IGNORE INTO posts_new
                (id, datetime_utc, source_name, author, text, likes, shares, comments)
            SELECT
                {sel_id},
                {sel_datetime},
                {sel_source},
                {sel_author},
                {sel_text},
                {sel_likes},
                {sel_shares},
                {sel_comments}
            FROM posts;
            """
        )

        con.execute("DROP TABLE posts;")
        con.execute("ALTER TABLE posts_new RENAME TO posts;")
        con.execute("CREATE INDEX IF NOT EXISTS ix_posts_datetime ON posts(datetime_utc DESC);")
        con.execute("CREATE INDEX IF NOT EXISTS ix_posts_source ON posts(source_name, datetime_utc DESC);")
        con.commit()
        log.warning("SQLite: migration complete.")
    except Exception as e:
        con.rollback()
        log.error("SQLite: migration failed: %s", e)
        raise


def ensure_schema(db_path: str | None = None) -> None:
    if db_path is not None:
        os.environ["SHAJARA_DB_PATH"] = db_path  # keep literal value, e.g. 'shajara.db'
    with connect() as con:
        _migrate_posts_if_needed(con)


def purge_empty_texts() -> int:
    """
    Delete only *true vacuum* rows:
      - text is NULL/blank AND author blank AND likes=shares=comments=0
    Everything else stays.
    """
    with connect() as con:
        cur = con.execute(
            """
            DELETE FROM posts
            WHERE (text IS NULL OR TRIM(text) = '')
              AND (author IS NULL OR TRIM(author) = '')
              AND COALESCE(likes,0)=0
              AND COALESCE(shares,0)=0
              AND COALESCE(comments,0)=0;
            """
        )
        con.commit()
        cnt = cur.rowcount or 0
        log.info("SQLite: purged %d true-vacuum rows.", cnt)
        return cnt


def _is_meaningful(text: str | None, author: str | None, likes, shares, comments) -> bool:
    """
    Lower-bar keeper:
      - keep if text has any non-whitespace
      - OR author non-empty
      - OR any engagement > 0
    """
    t = "" if text is None else str(text).strip()
    if t:
        return True
    a = "" if author is None else str(author).strip()
    if a:
        return True
    try:
        lk = int(likes or 0)
        sh = int(shares or 0)
        cm = int(comments or 0)
    except Exception:
        lk = sh = cm = 0
    return (lk + sh + cm) > 0


def upsert_posts(rows: Iterable[Sequence]) -> int:
    """
    rows must be tuples:
      (id: str, datetime_utc: str, source_name: str, author: str, text: str,
       likes: int, shares: int, comments: int)

    NOW: we only drop *true* vacuums (see _is_meaningful). Everything else stays.
    """
    prepared = []
    dropped = 0
    for r in rows:
        rid, dt, src, auth, txt, lk, sh, cm = r

        if not _is_meaningful(txt, auth, lk, sh, cm):
            dropped += 1
            continue

        rid = str(rid) if rid is not None else ""
        dt  = str(dt)  if dt  is not None else ""
        src = str(src) if src is not None else ""
        auth= "" if auth is None else str(auth)
        txt = "" if txt  is None else str(txt)

        try: lk = int(lk)
        except Exception: lk = 0
        try: sh = int(sh)
        except Exception: sh = 0
        try: cm = int(cm)
        except Exception: cm = 0

        prepared.append((rid, dt, src, auth, txt, lk, sh, cm))

    if not prepared:
        log.info("SQLite: nothing to upsert (dropped=%d).", dropped)
        return 0

    with connect() as con:
        cur = con.executemany(
            """
            INSERT INTO posts(id, datetime_utc, source_name, author, text, likes, shares, comments)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                datetime_utc=excluded.datetime_utc,
                source_name=excluded.source_name,
                author=excluded.author,
                text=excluded.text,
                likes=excluded.likes,
                shares=excluded.shares,
                comments=excluded.comments;
            """,
            prepared
        )
        con.commit()
        up = cur.rowcount or 0
        if dropped:
            log.info("SQLite: upserted %d rows, dropped %d true-vacuum rows.", up, dropped)
        else:
            log.info("SQLite: upserted %d rows.", up)
        return up
