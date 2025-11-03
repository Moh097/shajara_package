import re
from typing import List, Dict, Any, Optional

try:
    import tiktoken
except Exception:
    tiktoken = None

_AR_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)  # words (includes Arabic letters)
_WS = re.compile(r"\s+")

def simple_tokenize(text: str) -> List[str]:
    """
    Regex-based word tokenizer (unicode letters), language-agnostic.
    """
    if not text:
        return []
    return _AR_WORD.findall(text)

def token_stats(text: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute basic stats + optional GPT token count if tiktoken available.
    """
    tokens = simple_tokenize(text)
    s = {
        "words": len(tokens),
        "chars": len(text or ""),
        "preview": text[:120].replace("\n", " ") if text else "",
    }
    if tiktoken and model:
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        try:
            s["gpt_tokens"] = len(enc.encode(text or ""))
        except Exception:
            s["gpt_tokens"] = None
    else:
        s["gpt_tokens"] = None
    return s
