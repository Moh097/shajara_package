from __future__ import annotations
import os, json
from pathlib import Path

try:
    import tomllib  # py>=3.11
except Exception:
    tomllib = None

# Project root (shajara_package/)
_BASE_DIR = Path(__file__).resolve().parents[1]

# Defaults
DB_PATH = str(_BASE_DIR / "shajara.db")
OPENAI_MODEL = "gpt-4o-mini"

# Cache
_SECRETS = None

def _secrets_path() -> Path | None:
    # 1) explicit override
    p = os.getenv("STREAMLIT_SECRETS_PATH")
    if p and Path(p).exists():
        return Path(p)
    # 2) standard location
    candidate = _BASE_DIR / ".streamlit" / "secrets.toml"
    return candidate if candidate.exists() else None

def _load_streamlit_secrets() -> dict:
    global _SECRETS
    if _SECRETS is not None:
        return _SECRETS
    path = _secrets_path()
    if not path:
        _SECRETS = {}
        return _SECRETS
    if tomllib is None:
        # minimal TOML loader fallback (very permissive): treat as key="value" lines
        data: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip().strip('"').strip("'")
        _SECRETS = data
        return _SECRETS
    with path.open("rb") as f:
        _SECRETS = tomllib.load(f) or {}
    return _SECRETS

def S(key: str, default=None):
    """Read from env first, then secrets.toml, else default."""
    if key in os.environ:
        return os.environ[key]
    sec = _load_streamlit_secrets()
    # flat or top-level only
    if isinstance(sec, dict) and key in sec:
        v = sec[key]
        # normalize lists/dicts to JSON strings for consistency
        if isinstance(v, (list, dict)):
            return json.dumps(v, ensure_ascii=False)
        return str(v)
    return default

def export_secrets_to_env():
    """Copy secrets.toml into os.environ (strings only; lists/dicts JSON)."""
    sec = _load_streamlit_secrets()
    if not isinstance(sec, dict):
        return
    for k, v in sec.items():
        if isinstance(v, (list, dict)):
            os.environ[k] = json.dumps(v, ensure_ascii=False)
        elif v is None:
            continue
        else:
            os.environ[k] = str(v)

    # backfill defaults if still missing
    os.environ.setdefault("SHAJARA_DB_PATH", DB_PATH)
    os.environ.setdefault("OPENAI_MODEL", OPENAI_MODEL)
