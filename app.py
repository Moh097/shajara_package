# app.py
import os, re, io, json, time, math, subprocess
from typing import Dict, Any, List, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from modules.config import DB_PATH, using_azure
from modules.data_access import fetch_telegram_posts
from modules.tokenize import token_stats
from modules.llm import analyze_text_gpt
from modules.topic_merge import gpt_semantic_merge_terms
from utils.sqlite_client import ensure_schema

# Optional deps
try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

# ---- sanitize env (strip accidental quotes) ----
def _strip_env(name: str):
    v = os.getenv(name)
    if v:
        os.environ[name] = v.strip().strip('"').strip("'")
for _k in ("OPENAI_API_KEY","OPENAI_MODEL","AZURE_OPENAI_API_KEY","AZURE_OPENAI_ENDPOINT"):
    _strip_env(_k)

# ---- Arabic font config (robust on Windows/Linux/Mac) ----
def _register_ar_font() -> tuple[str | None, str | None]:
    env_path = (os.getenv("ARABIC_FONT_PATH") or "").strip().strip('"').strip("'")
    candidates_paths = [
        env_path,
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\Tahoma.ttf",
        r"C:\Windows\Fonts\times.ttf",
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for p in candidates_paths:
        if p and os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
                name = fm.FontProperties(fname=p).get_name()
                plt.rcParams["font.family"] = [name]
                plt.rcParams["axes.unicode_minus"] = False
                return p, name
            except Exception:
                pass
    candidate_families = [
        "Noto Naskh Arabic", "Noto Sans Arabic", "Traditional Arabic",
        "Tahoma", "Arial", "Times New Roman", "DejaVu Sans"
    ]
    for fam in candidate_families:
        try:
            fm.findfont(fm.FontProperties(family=fam), fallback_to_default=False)
            plt.rcParams["font.family"] = [fam]
            plt.rcParams["axes.unicode_minus"] = False
            return None, fam
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    return None, None

_AR_FONT_PATH, _AR_FONT_FAMILY = _register_ar_font()

def _wordcloud_font_path() -> str | None:
    if _AR_FONT_PATH and os.path.exists(_AR_FONT_PATH):
        return _AR_FONT_PATH
    try:
        fp = fm.findfont(fm.FontProperties(family=_AR_FONT_FAMILY), fallback_to_default=True)
        return fp if fp and os.path.exists(fp) else None
    except Exception:
        return None

def _shape_ar(s: str) -> str:
    try:
        if not re.search(r"[\u0600-\u06FF]", s or ""):
            return s or ""
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(s or ""))
    except Exception:
        return s or ""

# ================== STREAMLIT UI ==================
st.set_page_config(page_title="SHAJARA", layout="wide")
ensure_schema(DB_PATH)

# ---------- Sidebar ----------
st.sidebar.title("الإعدادات")
db_path = st.sidebar.text_input("مسار قاعدة البيانات (SQLite)", value=os.environ.get("SHAJARA_DB_PATH", DB_PATH))
if db_path and db_path != os.environ.get("SHAJARA_DB_PATH", DB_PATH):
    os.environ["SHAJARA_DB_PATH"] = db_path
    ensure_schema(db_path)

max_rows   = st.sidebar.slider("عدد السجلات", 50, 5000, 800, step=50)
search_txt = st.sidebar.text_input("بحث (LIKE)", value="")
since      = st.sidebar.text_input("من تاريخ (YYYY-MM-DD)", value="")
until      = st.sidebar.text_input("إلى تاريخ (YYYY-MM-DD)", value="")
limit_gpt  = st.sidebar.slider("حلّل أول N منشور", 10, 2000, 200, step=10)

# Action buttons
colb1, colb2, colb3 = st.sidebar.columns(3)
run_click   = colb1.button("تشغيل التحليل")
tg_click    = colb2.button("تشغيل جامع تيليجرام")
fb_click    = colb3.button("تشغيل جامع فيسبوك")

# One-shot run flag
if run_click:
    st.session_state["RUN_ANALYSIS_ONCE"] = True
run_analysis = st.session_state.pop("RUN_ANALYSIS_ONCE", False)

# Collectors
def _run_collector(script_rel: str):
    try:
        st.sidebar.info(f"تشغيل: {script_rel} ...")
        proc = subprocess.run(
            ["python", script_rel],
            capture_output=True, text=True, cwd=os.getcwd(), timeout=60*20
        )
        st.sidebar.success("اكتمل التنفيذ")
        with st.sidebar.expander("سجلّ التنفيذ"):
            st.sidebar.text(proc.stdout[-4000:] if proc.stdout else "(لا يوجد إخراج)")
            if proc.stderr:
                st.sidebar.text("\n[STDERR]\n" + proc.stderr[-2000:])
    except Exception as e:
        st.sidebar.error(f"فشل التنفيذ: {e}")

if tg_click: _run_collector("collectors/telegram_collector.py")
if fb_click: _run_collector("collectors/facebook_collector.py")

# ---------- Load data ----------
st.title("📡 شجرة — تحليل منشورات تيليجرام/فيسبوك")
with st.spinner("جلب البيانات من SQLite..."):
    df = fetch_telegram_posts(
        limit=max_rows,
        since_utc=(since.strip() + "T00:00:00Z") if since.strip() else None,
        until_utc=(until.strip() + "T23:59:59Z") if until.strip() else None,
        search=search_txt.strip() or None,
    )
if df.empty:
    st.warning("لا توجد بيانات. شغّل أحد المجمعين من الشريط الجانبي.")
    st.stop()
st.success(f"تم تحميل {len(df)} صفاً من {os.environ.get('SHAJARA_DB_PATH', 'shajara.db')}")

# ---------- Quick stats ----------
st.header("🧮 إحصاءات أساسية")
def _row_stats(row) -> Dict[str, Any]:
    s = token_stats((row.get("text") or ""), model=os.environ.get("OPENAI_MODEL"))
    s["id"] = row.get("id"); s["datetime_utc"] = row.get("datetime_utc"); s["source"] = row.get("source_name")
    return s

stat_df = pd.DataFrame(df.apply(_row_stats, axis=1).tolist())
c1,c2,c3,c4 = st.columns(4)
c1.metric("عدد المنشورات", len(df))
c2.metric("الوسيط (كلمات)", int(stat_df["words"].median()))
c3.metric("الوسيط (حروف)", int(stat_df["chars"].median()))
c4.metric("الوسيط (رموز GPT)", int(stat_df["gpt_tokens"].dropna().median()) if stat_df["gpt_tokens"].notna().any() else 0)

with st.expander("عرض الجدول"):
    st.dataframe(df[["datetime_utc","source_name","author","text","likes","shares","comments"]],
                 use_container_width=True, height=400)

fig = plt.figure()
stat_df["words"].plot(kind="hist", bins=30)
plt.title(_shape_ar("توزيع عدد الكلمات")); plt.xlabel(_shape_ar("كلمات")); plt.ylabel(_shape_ar("التكرار"))
st.pyplot(fig)

# ================== Helpers ==================
def parse_listlike(raw: Any) -> List[Any]:
    if raw is None: return []
    if isinstance(raw, list): return [x for x in raw if x is not None]
    if isinstance(raw, str):
        s = raw.strip()
        if not s: return []
        try:
            parsed = json.loads(s)
            return [x for x in parsed] if isinstance(parsed, list) else re.split(r"[;,/|]+", s)
        except Exception:
            return re.split(r"[;,/|]+", s)
    return [raw]

def extract_topics(x: Any) -> List[str]:
    return [str(t).strip() for t in parse_listlike(x) if str(t).strip()]

def extract_entities(x: Any) -> List[str]:
    out = []
    for v in parse_listlike(x):
        if isinstance(v, dict):
            val = v.get("text") or v.get("name") or v.get("entity") or v.get("label")
            if val: out.append(str(val))
        else:
            s = str(v).strip()
            if s: out.append(s)
    return out

def _get_client_and_model():
    if using_azure():
        from openai import AzureOpenAI
        cli = AzureOpenAI(
            api_key=(os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        )
        model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    else:
        from openai import OpenAI
        cli = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    return cli, model

def translate_list_to_ar(items: List[str]) -> Dict[str,str]:
    """Batch translate labels to Arabic. Returns mapping item->Arabic."""
    if not items: return {}
    cli, model = _get_client_and_model()
    out: Dict[str,str] = {}
    CHUNK = 80
    for i in range(0, len(items), CHUNK):
        chunk = [x for x in items[i:i+CHUNK] if x]
        if not chunk: continue
        prompt = {"task":"translate to Arabic surface labels only; return JSON object mapping each input to Arabic.",
                  "inputs": chunk}
        resp = cli.chat.completions.create(
            model=model, temperature=0,
            response_format={"type":"json_object"},
            messages=[
                {"role":"system","content":"أعد تسميات عربية موجزة فقط دون شرح."},
                {"role":"user","content":json.dumps(prompt, ensure_ascii=False)}
            ],
        )
        txt = resp.choices[0].message.content if resp and resp.choices else "{}"
        try:
            got = json.loads(txt)
            if isinstance(got, dict):
                out.update({k:str(v) for k,v in got.items()})
        except Exception:
            out.update({x:x for x in chunk})
    return out

def map_sentiment_ar(x: str) -> str:
    if not x: return ""
    m = {"positive":"إيجابي","negative":"سلبي","neutral":"محايد"}
    return m.get(str(x).strip().lower(), str(x))

def map_tension_ar(x: str) -> str:
    if not x: return ""
    m = {"low":"منخفض","medium":"متوسط","high":"مرتفع"}
    return m.get(str(x).strip().lower(), str(x))

# ================== RUN ANALYSIS ==================
st.header("🧠 التحليل (المواضيع، الكيانات، الملخص، المشاعر)")

if run_analysis:
    # Initialize caches on MAIN thread (safe)
    if "gpt_cache" not in st.session_state:
        st.session_state["gpt_cache"] = {}
    if "ar_translate_cache" not in st.session_state:
        st.session_state["ar_translate_cache"] = {}

    local_cache = dict(st.session_state["gpt_cache"])  # thread-safe copy
    cache_lock = Lock()

    results: List[Dict] = []
    progress = st.progress(0)
    N = min(int(limit_gpt), len(df))

    def worker(i: int, text: str) -> Dict[str, Any]:
        key = f"{os.environ.get('OPENAI_MODEL','gpt')}|{hash(text or '')}"
        with cache_lock:
            cached = local_cache.get(key)
        if cached is not None:
            return cached
        try:
            res = analyze_text_gpt(text or "")
        except Exception as e:
            res = {"error": str(e)}
        with cache_lock:
            local_cache[key] = res
        return res

    # Run GPT per-post in parallel (no session_state inside threads)
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(worker, i, df.iloc[i].get("text")): i for i in range(N)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]; res = fut.result()
            row = df.iloc[i].to_dict()
            results.append({
                "id":row.get("id"),
                "datetime_utc":row.get("datetime_utc"),
                "source_name":row.get("source_name"),
                "author":row.get("author"),
                **res
            })
            done += 1
            progress.progress(int(done/N*100))
    progress.empty()

    # Sync cache back on MAIN thread
    st.session_state["gpt_cache"] = local_cache

    res_df = pd.DataFrame(results)
    st.subheader("النتائج")
    st.dataframe(res_df, use_container_width=True, height=360)

    # Collect raw topics/entities
    all_topics_raw, all_entities_raw = [], []
    for r in results:
        all_topics_raw += extract_topics(r.get("topics"))
        all_entities_raw += extract_entities(r.get("entities"))

    # Translate to Arabic (cached, main thread)
    to_translate = [t for t in set(all_topics_raw + all_entities_raw)
                    if t not in st.session_state["ar_translate_cache"]]
    if to_translate:
        st.session_state["ar_translate_cache"].update(translate_list_to_ar(to_translate))
    T = st.session_state["ar_translate_cache"]

    # Arabic normalized uniq lists
    def to_ar(t: str) -> str: return T.get(t, t)
    uniq_topics = sorted({to_ar(t).strip().lower() for t in all_topics_raw if to_ar(t)})
    uniq_entities = sorted({to_ar(e).strip().lower() for e in all_entities_raw if to_ar(e)})

    # Semantic merge (Arabic terms)
    topic_map = gpt_semantic_merge_terms(uniq_topics) if uniq_topics else {}
    if not topic_map: topic_map = {k:k for k in uniq_topics}
    ent_map = gpt_semantic_merge_terms(uniq_entities) if uniq_entities else {}
    if not ent_map: ent_map = {k:k for k in uniq_entities}

    # Rewrite rows with Arabic + merged
    canon_results: List[Dict[str,Any]] = []
    for r in results:
        topics_ar = [to_ar(t).strip().lower() for t in extract_topics(r.get("topics")) if to_ar(t)]
        entities_ar = [to_ar(e).strip().lower() for e in extract_entities(r.get("entities")) if to_ar(e)]
        topics_ar = list(dict.fromkeys([topic_map.get(t, t) for t in topics_ar]))
        entities_ar = list(dict.fromkeys([ent_map.get(e, e) for e in entities_ar]))
        row = dict(r)
        row["topics"] = topics_ar
        row["entities"] = entities_ar
        row["sentiment"] = map_sentiment_ar(r.get("sentiment") or "")
        row["tension"]   = map_tension_ar(r.get("tension") or "")
        if r.get("summary"):
            smap = translate_list_to_ar([str(r["summary"])])
            row["summary"] = smap.get(str(r["summary"]), str(r["summary"]))
        canon_results.append(row)

    res_df = pd.DataFrame(canon_results)
    st.subheader("النتائج بعد التوحيد بالعربية")
    st.dataframe(res_df, use_container_width=True, height=360)

    # Sentiment chart
    if "sentiment" in res_df.columns and res_df["sentiment"].notna().any():
        counts = res_df["sentiment"].value_counts(dropna=True)
        fig2 = plt.figure()
        counts.index = [_shape_ar(i) for i in counts.index]
        counts.plot(kind="bar")
        plt.title(_shape_ar("توزيع المشاعر")); plt.xlabel(_shape_ar("التصنيف")); plt.ylabel(_shape_ar("العدد"))
        st.pyplot(fig2)

    # Tension chart
    if "tension" in res_df.columns and res_df["tension"].notna().any():
        counts = res_df["tension"].value_counts(dropna=True)
        fig3 = plt.figure()
        counts.index = [_shape_ar(i) for i in counts.index]
        counts.plot(kind="bar")
        plt.title(_shape_ar("مستويات التوتر")); plt.xlabel(_shape_ar("التصنيف")); plt.ylabel(_shape_ar("العدد"))
        st.pyplot(fig3)

    # Top topics
    all_topics = []
    if "topics" in res_df.columns:
        for tlist in res_df["topics"].dropna():
            if isinstance(tlist, list):
                all_topics.extend([t for t in tlist if t])
    if all_topics:
        topics_df = pd.DataFrame({"topic": all_topics})
        top_topics = topics_df["topic"].value_counts().head(25)
        fig4 = plt.figure()
        top_topics.index = [_shape_ar(i) for i in top_topics.index]
        top_topics.plot(kind="bar")
        plt.title(_shape_ar("أكثر المواضيع تداولاً")); plt.xlabel(_shape_ar("الموضوع")); plt.ylabel(_shape_ar("العدد"))
        st.pyplot(fig4)

    # Build graph + wordcloud in parallel
    def build_graph_html():
        from modules.viz import build_cooccurrence_graph
        return build_cooccurrence_graph(canon_results, top_k_terms=25, min_edge_weight=2)

    def build_wordcloud_png():
        if WordCloud is None:
            return None
        freq = Counter()
        for row in canon_results:
            for e in row.get("entities") or []:
                if e: freq[e] += 1
        if not freq:
            return None
        items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:300]
        shaped = {_shape_ar(k): v for k, v in items}
        wc = WordCloud(
            width=1300, height=550, background_color="white",
            max_words=300, collocations=False, prefer_horizontal=1.0,
            regexp=r"[\u0600-\u06FF\w-]+",
            font_path=_wordcloud_font_path(),
        ).generate_from_frequencies(shaped)
        buf = io.BytesIO(); wc.to_image().save(buf, format="PNG"); buf.seek(0)
        return buf

    with ThreadPoolExecutor(max_workers=4) as ex:
        f_graph = ex.submit(build_graph_html)
        f_wc    = ex.submit(build_wordcloud_png)

        st.subheader("الرسم الشبكي لتشارك المواضيع")
        graph_html = f_graph.result()
        st.components.v1.html(graph_html, height=680, scrolling=True)

        st.subheader("سحابة الكلمات — الكيانات")
        wc_buf = f_wc.result()
        if wc_buf is None:
            st.info("لا توجد كيانات كافية أو حزمة wordcloud غير مثبتة.")
        else:
            img = wc_buf.getvalue()
            # FIX: use_container_width instead of deprecated use_column_width
            st.image(img, use_container_width=True)
            st.download_button("تحميل صورة السحابة (PNG)", data=img, file_name="entities_wordcloud.png", mime="image/png")

    # Export
    st.download_button(
        "تنزيل النتائج (CSV)",
        data=res_df.to_csv(index=False).encode("utf-8"),
        file_name="gpt_analysis_results_ar.csv",
        mime="text/csv",
    )
else:
    st.info("اضغط زر **تشغيل التحليل** من الشريط الجانبي.")
