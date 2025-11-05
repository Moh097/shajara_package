# modules/llm.py
from __future__ import annotations
import json
from typing import Dict, Any
from openai import OpenAI

from .config import S, OPENAI_MODEL

def _client_and_model():
    cli = OpenAI(api_key=S("OPENAI_API_KEY"))
    model = S("OPENAI_MODEL", OPENAI_MODEL)
    return cli, model

def analyze_text_gpt(text: str) -> Dict[str, Any]:
    """
    Return a dict with keys:
      - topics: List[str] (short English tags)
      - entities: List[str] (plain English surface forms)
      - summary: str (<= 3 sentences, English)
      - sentiment: one of {"positive","neutral","negative"}
      - tension: one of {"low","medium","high"}  (heuristic "intensity" scale)
    All outputs MUST be English.
    """
    cli, model = _client_and_model()
    payload = {
        "instruction": (
            "You are an information extraction assistant. "
            "Extract concise English topics (as short tags), named entities (surface forms only), "
            "a brief English summary (max 3 sentences), overall sentiment (positive/neutral/negative), "
            "and an intensity scale called 'tension' (low/medium/high). "
            "Return a strict JSON object with keys exactly: topics, entities, summary, sentiment, tension. "
            "Do not include any explanations."
        ),
        "input": (text or "")[:10000]  # safety cap
    }
    resp = cli.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Respond in English only. Output must be valid JSON."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ],
    )
    raw = resp.choices[0].message.content if resp and resp.choices else "{}"
    try:
        data = json.loads(raw)
    except Exception:
        data = {}

    # Normalize structure
    topics = data.get("topics") or []
    if isinstance(topics, str): topics = [topics]
    entities = data.get("entities") or []
    if isinstance(entities, str): entities = [entities]
    out = {
        "topics": [str(t).strip() for t in topics if str(t).strip()],
        "entities": [str(e).strip() for e in entities if str(e).strip()],
        "summary": str(data.get("summary") or "").strip(),
        "sentiment": (str(data.get("sentiment") or "").strip().lower() or "neutral"),
        "tension": (str(data.get("tension") or "").strip().lower() or "medium"),
    }
    # Guard sentiment/tension values
    if out["sentiment"] not in {"positive","neutral","negative"}:
        out["sentiment"] = "neutral"
    if out["tension"] not in {"low","medium","high"}:
        out["tension"] = "medium"
    return out
