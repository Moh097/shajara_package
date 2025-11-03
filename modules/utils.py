import hashlib
from typing import Dict, Any

def row_key(text: str, model: str) -> str:
    h = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    return f"{model}:{h}"

def safe_get(d: Dict[str, Any], k: str, default=None):
    v = d.get(k)
    return default if v is None else v
