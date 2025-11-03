# modules/config.py
from __future__ import annotations
import os
import pathlib
from typing import Any, Dict

def _load_streamlit_secrets() -> Dict[str, Any]:
    # Try Streamlit runtime
    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "secrets"):
            try:
                return dict(st.secrets)
            except Exception:
                return {k: st.secrets[k] for k in st.secrets}
    except Exception:
        pass
    # Try local .streamlit/secrets.toml for non-Streamlit runs
    try:
        import tomllib  # py311+
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
    return v.strip().strip('"').strip("'") if isinstance(v, str) else v

def S(key: str, default: Any = None) -> Any:
    """Secrets > env > default (strings sanitized)."""
    if key in _SECRETS:
        return _sanitize(_SECRETS[key])
    envv = os.getenv(key)
    return _sanitize(envv) if envv is not None else default

def export_secrets_to_env(keys: list[str] | None = None) -> None:
    """Expose secrets to os.environ so subprocesses inherit them."""
    if not _SECRETS:
        return
    for k in (keys or list(_SECRETS.keys())):
        if os.getenv(k) is None and k in _SECRETS and not isinstance(_SECRETS[k], (dict, list)):
            os.environ[k] = str(_SECRETS[k])

# No Azure. Only OpenAI.
OPENAI_MODEL: str = S("OPENAI_MODEL", "gpt-4o-mini")
DB_PATH: str = S("SHAJARA_DB_PATH", "shajara.db")
