# modules/topic_merge.py
from __future__ import annotations
import json
from typing import Dict, List
from .config import S, OPENAI_MODEL

def _get_client_and_model():
    from openai import OpenAI
    cli = OpenAI(api_key=S("OPENAI_API_KEY"))
    model = S("OPENAI_MODEL", OPENAI_MODEL)
    return cli, model

_PROMPT = (
    "You will receive a list of short labels (topics/entities). "
    "Return a JSON object mapping EACH input label to a canonical merged label. "
    "Merge near-duplicates, spell variants, plural/singular, Arabic/English equivalents, "
    "and case/diacritic differences. Keep canonical labels short and human-friendly. "
    "Keys must be original inputs (exact), values are their chosen canonical form. "
    "Output ONLY a JSON object."
)

def gpt_semantic_merge_terms(terms: List[str], batch_size: int = 120) -> Dict[str, str]:
    """
    Map each input term to a canonical label using a small LLM pass.
    Safe: chunks inputs; falls back to identity mapping on any error.
    """
    cli, model = _get_client_and_model()
    # Clean + stable order
    seen = set()
    items = []
    for t in terms or []:
        t = (t or "").strip()
        if t and t not in seen:
            seen.add(t); items.append(t)
    if not items:
        return {}

    out: Dict[str, str] = {}
    for i in range(0, len(items), batch_size):
        chunk = items[i:i+batch_size]
        try:
            resp = cli.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _PROMPT},
                    {"role": "user", "content": json.dumps(chunk, ensure_ascii=False)},
                ],
            )
            txt = resp.choices[0].message.content if resp and resp.choices else "{}"
            data = json.loads(txt or "{}")
            if isinstance(data, dict):
                for k, v in data.items():
                    k2 = str(k).strip()
                    v2 = str(v).strip()
                    if k2:
                        out[k2] = v2 if v2 else k2
            else:
                # Fallback: identity
                for k in chunk:
                    out[k] = k
        except Exception:
            for k in chunk:
                out[k] = k
    return out
