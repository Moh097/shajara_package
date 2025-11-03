# modules/tokenize.py
from __future__ import annotations
import re
from typing import Dict, Any

# Try to import tiktoken; fall back if missing
try:
    import tiktoken  # type: ignore
    _HAS_TIKTOKEN = True
except Exception:
    tiktoken = None
    _HAS_TIKTOKEN = False

# Map models to encodings (proxy for OpenAI models)
_MODEL_TO_ENCODING = {
    None: "cl100k_base",
    "gpt-4o": "cl100k_base",
    "gpt-4o-mini": "cl100k_base",
    "gpt-4o-mini-2024-07-18": "cl100k_base",
}

_WORD_RE = re.compile(r"\w+|\S", re.UNICODE)

def _estimate_tokens(text: str) -> int:
    """
    Fallback: ~1 token per 4 chars blended with word count.
    This is good enough for charts when tiktoken isn't available.
    """
    if not text:
        return 0
    approx = max(1, int(len(text) / 4))
    words = len(_WORD_RE.findall(text))
    return int(0.6 * approx + 0.4 * words)

def token_stats(text: str, model: str | None = None) -> Dict[str, Any]:
    text = text or ""
    chars = len(text)
    words = len(_WORD_RE.findall(text))
    if _HAS_TIKTOKEN:
        enc_name = _MODEL_TO_ENCODING.get((model or "").strip() or None, "cl100k_base")
        try:
            enc = tiktoken.get_encoding(enc_name)
            toks = len(enc.encode(text))
        except Exception:
            toks = _estimate_tokens(text)
    else:
        toks = _estimate_tokens(text)
    return {
        "chars": chars,
        "words": words,
        "gpt_tokens": toks,
    }
