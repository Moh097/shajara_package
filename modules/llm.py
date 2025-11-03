from __future__ import annotations
import json, time, hashlib
from typing import Dict, Any, List

from .config import (
    OPENAI_API_KEY, OPENAI_MODEL,
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT,
    using_azure
)

# Lazy import to keep Streamlit fast on non-LLM views
_client = None

def _client_openai():
    global _client
    if _client:
        return _client

    if using_azure():
        # Azure OpenAI via "OpenAI" client (new SDK)
        from openai import AzureOpenAI
        _client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2024-08-01-preview",
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
    else:
        from openai import OpenAI
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

def _model_name():
    return AZURE_OPENAI_DEPLOYMENT if using_azure() else OPENAI_MODEL

def analyze_text_gpt(text: str) -> Dict[str, Any]:
    """
    Calls GPT to return JSON with:
      sentiment: positive|negative|neutral
      tension: low|medium|high
      topics: [..]
      entities: [{text, type}]
      summary: string (max ~50 words)
      language: iso639-1 (guess)
    """
    if not text or not text.strip():
        return {}

    client = _client_openai()
    model = _model_name()

    sys_prompt = (
        "You are a concise analyst for Arabic and English Telegram posts. "
        "Return STRICT JSON with keys: sentiment, tension, topics, entities, summary, language. "
        "sentiment ∈ {positive, negative, neutral}; tension ∈ {low, medium, high}. "
        "topics: <=5 single- or bi-gram strings. entities: <=6 items with {text,type}. "
        "summary: <=50 words. language: iso639-1 code."
    )

    user_prompt = f"Analyze the following post:\n---\n{text}\n---"

    # Prefer JSON mode if available
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        content = resp.choices[0].message.content
    except Exception:
        # Fallback: ask for fenced JSON
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": sys_prompt + " If not in JSON mode, return only a minified JSON object."},
                {"role": "user", "content": user_prompt}
            ],
        )
        content = resp.choices[0].message.content

    # Parse JSON
    try:
        data = json.loads(content)
    except Exception:
        # last resort: try to extract {...}
        import re
        m = re.search(r"\{.*\}", content, re.S)
        data = json.loads(m.group(0)) if m else {}

    # light sanitization
    out = {
        "sentiment": (data.get("sentiment") or "").lower()[:8],
        "tension": (data.get("tension") or "").lower()[:6],
        "topics": data.get("topics") or [],
        "entities": data.get("entities") or [],
        "summary": data.get("summary") or "",
        "language": (data.get("language") or "").lower()[:5],
    }
    return out
