# modules/topic_merge.py
from __future__ import annotations
import json, os, re, time
from typing import Dict, List
from modules.config import using_azure, OPENAI_MODEL, AZURE_OPENAI_DEPLOYMENT

def _get_client_and_model():
    if using_azure():
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=(os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        )
        model = os.getenv("OPENAI_TOPIC_MODEL") or AZURE_OPENAI_DEPLOYMENT or OPENAI_MODEL
    else:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("OPENAI_TOPIC_MODEL") or os.getenv("OPENAI_MODEL", OPENAI_MODEL)
    return client, model

def _extract_json(text: str) -> dict:
    if not text: return {}
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except Exception: return {}
    return {}

_SYS_PROMPT = (
    "أنت مسؤول عن توحيد تسميات المواضيع والكيانات بالعربية.\n"
    "عند إعطائك مصطلحاً جديداً NEW_TERM وقائمة تسميات قياسية EXISTING، قرّر هل المصطلح مكافئ دلالياً لإحداها "
    "(بما في ذلك اختلافات الوصف مثل مدينة/محافظة/منطقة أو city/province). إذا كان مكافئاً: action='merge' و canonical يجب أن يكون "
    "عنصراً مطابقاً 100% من EXISTING. إن لم يكن مكافئاً: action='new' واقترح تسمية قياسية عربية موجزة.\n"
    "أعد **فقط** JSON بهذا الشكل: {\"action\":\"merge\"|\"new\",\"canonical\":\"...\"}"
)

def _gpt_decide_merge(client, model: str, new_term: str, existing: List[str]) -> Dict[str, str]:
    max_candidates = int(os.getenv("SEMANTIC_MERGE_MAX_CANDIDATES", "60"))
    candidates = existing[-max_candidates:] if max_candidates > 0 else existing
    payload = {"NEW_TERM": new_term, "EXISTING": candidates}

    resp = client.chat.completions.create(
        model=model, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role":"system","content":_SYS_PROMPT},
                  {"role":"user","content":json.dumps(payload, ensure_ascii=False)}],
    )
    text = resp.choices[0].message.content if resp and resp.choices else ""
    data = _extract_json(text)
    action = (data.get("action") or "").strip().lower()
    canonical = (data.get("canonical") or "").strip()
    if action == "merge" and canonical in candidates:
        return {"action":"merge","canonical":canonical}
    if action == "new" and canonical:
        return {"action":"new","canonical":canonical}
    return {"action":"new","canonical":new_term}

def gpt_semantic_merge_terms(terms: List[str], sleep_sec: float = 0.05) -> Dict[str, str]:
    client, model = _get_client_and_model()
    mapping: Dict[str, str] = {}
    canon_list: List[str] = []
    for term in terms:
        t = (term or "").strip()
        if not t or t in mapping: continue
        try:
            decision = _gpt_decide_merge(client, model, t, canon_list)
        except Exception:
            decision = {"action":"new","canonical":t}
        action = decision.get("action"); canonical = decision.get("canonical") or t
        if action == "merge":
            mapping[t] = canonical
        else:
            canon_list.append(canonical)
            mapping[t] = canonical
        if sleep_sec: time.sleep(sleep_sec)
    return mapping
