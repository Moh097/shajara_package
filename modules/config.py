# modules/config.py
from __future__ import annotations
import os
import pathlib
from typing import Any, Dict

def _load_streamlit_secrets() -> Dict[str, Any]:
    """Load secrets from streamlit runtime if available, else from .streamlit/secrets.toml."""
    # 1) Try streamlit runtime
    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "secrets"):
            try:
                return dict(st.secrets)  # Config object -> dict
            except Exception:
                # Fallback: iterate keys
                return {k: st.secrets[k] for k in st.secrets}
    except Exception:
        pass

    # 2) Try reading .streamlit/secrets.toml directly (outside Streamlit)
    try:
        import tomllib  # Python 3.11+
        # Search CWD then project root (parent of modules/)
        candidates = [
            pathlib.Path.cwd() / ".streamlit" / "secrets.toml",
            pathlib.Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml",
        ]
        for p in candidates:
            if p.is_file():
                return tomllib.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass

    return {}

_SECRETS: Dict[str, Any] = _load_streamlit_secrets()

def _sanitize(v: Any) -> Any:
    if isinstance(v, str):
        return v.strip().strip('"').strip("'")
    return v

def S(key: str, default: Any = None) -> Any:
    """
    Get a value with precedence: Streamlit secrets -> env var -> default.
    Strings are stripped of extra quotes/spaces; non-strings returned as-is.
    """
    if key in _SECRETS:
        return _sanitize(_SECRETS[key])
    envv = os.getenv(key, None)
    if envv is not None:
        return _sanitize(envv)
    return default

def export_secrets_to_env(keys: list[str] | None = None) -> None:
    """
    Export secrets into os.environ so child processes (collectors) inherit them.
    Does not overwrite existing env vars.
    """
    if not _SECRETS:
        return
    ks = keys or list(_SECRETS.keys())
    for k in ks:
        if os.getenv(k) is None and k in _SECRETS:
            v = _SECRETS[k] 
            if not isinstance(v, (dict, list)):
                os.environ[k] = str(v)

def using_azure() -> bool:
    return bool(S("AZURE_OPENAI_ENDPOINT"))

# Common config values (evaluated at import)
DB_PATH: str = S("SHAJARA_DB_PATH", "shajara.db")
OPENAI_MODEL: str = S("OPENAI_MODEL", "gpt-4o-mini")

# Telegram (if needed elsewhere)
TELEGRAM_API_ID = S("TELEGRAM_API_ID")
TELEGRAM_API_HASH = S("TELEGRAM_API_HASH")
TELEGRAM_STRING_SESSION = S("TELEGRAM_STRING_SESSION")
