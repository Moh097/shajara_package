# utils/supabase_client.py

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests

from modules.config import S

log = logging.getLogger("supabase_client")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get_supabase_config() -> Tuple[str, str]:
    """
    Returns (url, key) from env or .streamlit/secrets.toml.

    Supports:
      - SUPABASE_URL
      - SUPABASE_SERVICE_ROLE_KEY (preferred if present)
      - SUPABASE_ANON_KEY        (fallback – what you're using now)
    """
    url = (S("SUPABASE_URL", os.environ.get("SUPABASE_URL", "")) or "").rstrip("/")
    key = (
        S("SUPABASE_SERVICE_ROLE_KEY", os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""))  # if you ever add it
        or S("SUPABASE_ANON_KEY", os.environ.get("SUPABASE_ANON_KEY", ""))               # currently used
        or ""
    )
    if not url or not key:
        raise RuntimeError(
            "Supabase config missing: SUPABASE_URL or "
            "SUPABASE_SERVICE_ROLE_KEY / SUPABASE_ANON_KEY not set "
            "(env or .streamlit/secrets.toml)."
        )
    return url, key


def _base_headers(*, count: bool = False) -> Dict[str, str]:
    _, key = _get_supabase_config()
    headers: Dict[str, str] = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }
    if count:
        # Ask PostgREST for exact total using Content-Range
        headers["Prefer"] = "count=exact"
    return headers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select(
    table: str,
    params: Dict[str, Any],
    *,
    count: bool = False,
    timeout: int = 30,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """
    Generic SELECT wrapper around Supabase REST (PostgREST).
    """
    base_url, _ = _get_supabase_config()
    url = f"{base_url}/rest/v1/{table}"

    log.debug("Supabase SELECT %s params=%s", table, params)
    resp = requests.get(
        url,
        headers=_base_headers(count=count),
        params=params,
        timeout=timeout,
    )
    resp.raise_for_status()

    total: Optional[int] = None
    if count:
        cr = resp.headers.get("Content-Range")
        if cr and "/" in cr:
            # Content-Range: 0-0/1234
            try:
                total = int(cr.split("/")[-1])
            except ValueError:
                total = None

    try:
        data = resp.json()
    except ValueError:
        data = []

    if isinstance(data, dict):
        data = [data]

    return data, total


def upsert(
    table: str,
    rows: List[Dict[str, Any]],
    *,
    timeout: int = 60,
) -> int:
    """
    Generic UPSERT (insert/update) into Supabase.

    Used by collectors to push rows into `posts`.
    Returns the number of rows Supabase reports back.
    """
    if not rows:
        return 0

    base_url, _ = _get_supabase_config()
    url = f"{base_url}/rest/v1/{table}"

    headers = _base_headers()
    headers.update(
        {
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=representation",
        }
    )

    payload = json.dumps(rows, ensure_ascii=False)
    log.debug("Supabase UPSERT %s rows=%d", table, len(rows))

    resp = requests.post(
        url,
        headers=headers,
        data=payload.encode("utf-8"),
        timeout=timeout,
    )
    resp.raise_for_status()

    try:
        data = resp.json()
    except ValueError:
        data = []

    if isinstance(data, list):
        return len(data)
    return len(rows)
