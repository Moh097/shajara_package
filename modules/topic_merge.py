# modules/topic_merge.py
from __future__ import annotations
import json
from typing import Dict, List
from openai import OpenAI
from .config import S, OPENAI_MODEL

def _client_and_model():
    cli = OpenAI(api_key=S("OPENAI_API_KEY"))
    model = S("OPENAI_MODEL", OPENAI_MODEL)
    return cli, model

def gpt_semantic_merge_terms(terms: List[str]) -> Dict[str, str]:
    """
    Given a list of English terms, return a mapping {original -> canonical}
    that merges duplicates/near-duplicates/synonyms into a concise English canonical label.
    If the list is large, call in chunks and merge the results.
    """
    terms = [t for t in (terms or []) if t and t.strip()]
    if not terms:
        return {}

    cli, model = _client_and_model()
    CHUNK = 120
    result: Dict[str, str] = {}
    for i in range(0, len(terms), CHUNK):
        chunk = terms[i:i+CHUNK]
        prompt = {
            "instruction": (
                "Given a list of English terms (topics/entities), group duplicates and near-duplicates, "
                "and map each original term to a concise English canonical label. "
                "Return a JSON object where keys are the original terms and values are the canonical labels. "
                "Do not explain."
            ),
            "terms": chunk
        }
        resp = cli.chat.completions.create(
            model=model, temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Respond in English only. Output must be valid JSON."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
            ],
        )
        txt = resp.choices[0].message.content if resp and resp.choices else "{}"
        try:
            got = json.loads(txt)
            if isinstance(got, dict):
                # Only keep mappings for seen chunk terms
                for k, v in got.items():
                    if k in chunk and isinstance(v, str) and v.strip():
                        result[k] = v.strip()
        except Exception:
            # Fallback: identity mapping for this chunk
            for k in chunk:
                result.setdefault(k, k)
    # Ensure every term is present at least as identity
    for t in terms:
        result.setdefault(t, t)
    return result
