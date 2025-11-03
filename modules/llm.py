# modules/llm.py
from __future__ import annotations
import json
from typing import Any, Dict, List
from .config import S, OPENAI_MODEL

def _get_client_and_model():
    from openai import OpenAI
    cli = OpenAI(api_key=S("OPENAI_API_KEY"))
    model = S("OPENAI_MODEL", OPENAI_MODEL)
    return cli, model

_SYSTEM = (
    "You are an NLP microservice. "
    "Return a compact JSON object with keys exactly: "
    'summary (string), topics (array of short strings), entities (array of strings), '
    'sentiment ("positive"|"negative"|"neutral"), tension ("low"|"medium"|"high"). '
    "No extra commentary."
)

def _empty() -> Dict[str, Any]:
    return {"summary": "", "topics": [], "entities": [], "sentiment": "", "tension": ""}

def analyze_text_gpt(text: str) -> Dict[str, Any]:
    if not (text or "").strip():
        return _empty()
    cli, model = _get_client_and_model()
    try:
        resp = cli.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": (text or "")[:8000]},
            ],
        )
        content = resp.choices[0].message.content if resp and resp.choices else "{}"
        data = json.loads(content or "{}")
    except Exception as e:
        out = _empty(); out["error"] = str(e); return out

    summary = data.get("summary") or ""
    topics = data.get("topics") or []
    entities = data.get("entities") or []
    sentiment = (data.get("sentiment") or "").strip().lower()
    tension = (data.get("tension") or "").strip().lower()

    if isinstance(topics, str): topics = [topics]
    if isinstance(entities, str): entities = [entities]

    def _dedup(seq: List[str]) -> List[str]:
        seen, out = set(), []
        for s in seq:
            s = (s or "").strip()
            if s and s not in seen:
                seen.add(s); out.append(s)
        return out

    return {
        "summary": str(summary),
        "topics": _dedup([str(t) for t in topics]),
        "entities": _dedup([str(e) for e in entities]),
        "sentiment": sentiment if sentiment in {"positive", "negative", "neutral"} else "",
        "tension": tension if tension in {"low", "medium", "high"} else "",
    }
