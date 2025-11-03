# utils/supabase_client.py
# Supabase REST helper with numeric coercion + (platform,platform_post_id) conflict
import os, time, json, requests

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

HEADERS = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

_NUMERIC_FIELDS = {"shares", "likes", "comments"}

def _to_int_or_none(v):
    if v is None:
        return None
    if isinstance(v, int):
        return v
    try:
        s = str(v)
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else None
    except Exception:
        return None

def _clean_row(row: dict) -> dict:
    out = {}
    for k, v in (row or {}).items():
        if k in _NUMERIC_FIELDS:
            out[k] = _to_int_or_none(v)
            continue
        if v is None:
            out[k] = None
            continue
        try:
            json.dumps(v)
            out[k] = v
        except Exception:
            out[k] = str(v)
    return out

def upsert_posts(rows, batch_size=50, max_retries=3, on_conflict="platform,platform_post_id"):
    if not rows:
        return
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY are required.")

    url = f"{SUPABASE_URL}/rest/v1/posts"
    params = {"on_conflict": on_conflict}
    clean_rows = [_clean_row(r) for r in rows]

    for i in range(0, len(clean_rows), batch_size):
        batch = clean_rows[i:i+batch_size]
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = requests.post(url, params=params, headers=HEADERS, data=json.dumps(batch), timeout=60)
                if resp.status_code in (200, 201):
                    break
                if resp.status_code == 409:
                    # fall back row-by-row to skip dupes
                    for row in batch:
                        r1 = requests.post(url, params=params, headers=HEADERS, data=json.dumps([row]), timeout=60)
                        if r1.status_code not in (200, 201, 409):
                            raise Exception(f"Row insert failed: {r1.status_code} {r1.text}")
                    break
                if attempt < max_retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise Exception(f"Batch insert failed after retries: {resp.status_code} {resp.text}")
            except requests.RequestException as e:
                if attempt < max_retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise e
