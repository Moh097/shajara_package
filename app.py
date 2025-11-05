# app.py
import os, re, io, json, subprocess, hashlib, sys, time, sqlite3
from typing import Dict, Any, List
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path
from contextlib import contextmanager

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from modules.config import DB_PATH, OPENAI_MODEL, S, export_secrets_to_env
from modules.data_access import fetch_telegram_posts
from modules.tokenize import token_stats
from modules.llm import analyze_text_gpt
from modules.topic_merge import gpt_semantic_merge_terms
from utils.sqlite_client import ensure_schema, purge_empty_texts

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

# ---------- Prime env ----------
export_secrets_to_env()

# ---------- Fonts ----------
def _register_font() -> tuple[str | None, str | None]:
    candidates_paths = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\Tahoma.ttf",
        r"C:\Windows\Fonts\times.ttf",
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
    for fam in ["Arial", "Tahoma", "Times New Roman", "DejaVu Sans", "Helvetica", "Liberation Sans"]:
        try:
            fm.findfont(fm.FontProperties(family=fam), fallback_to_default=False)
            plt.rcParams["font.family"] = [fam]
            plt.rcParams["axes.unicode_minus"] = False
            return None, fam
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    return None, None

_FONT_PATH, _FONT_FAMILY = _register_font()

# ---------- App skeleton ----------
st.set_page_config(page_title="SHAJARA", layout="wide")
BASE_DIR = Path(__file__).resolve().parent  # .../shajara_package

# Force shared DB name for app + collectors; no UI for DB path.
os.environ.setdefault("SHAJARA_DB_PATH", "shajara.db")
ensure_schema(os.environ["SHAJARA_DB_PATH"])
try:
    purge_empty_texts()
except Exception:
    pass

# ---------- Helpers ----------
def _to_pos_int(s, default=10) -> int:
    try:
        v = int(str(s).strip())
        return v if v > 0 else default
    except Exception:
        return default

def _parse_list_from_str_or_json(raw: str | None) -> List[str]:
    if not raw: return []
    raw = raw.strip()
    if not raw: return []
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    return [p.strip() for p in raw.replace("\r", "").replace("\n", ",").split(",") if p.strip()]

def _load_initial_channels() -> List[str]:
    override_path = BASE_DIR / ".streamlit" / "channels.json"
    if override_path.exists():
        try:
            data = json.loads(override_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except Exception:
            pass
    return _parse_list_from_str_or_json(S("TELEGRAM_CHANNELS", "") or "")

def _load_initial_search_terms() -> List[str]:
    return _parse_list_from_str_or_json(S("TELEGRAM_SEARCH_TERMS", "") or "")

def _save_channels_locally(chs: List[str]) -> None:
    p = BASE_DIR / ".streamlit" / "channels.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(chs, ensure_ascii=False, indent=2), encoding="utf-8")

def _st_image(img_bytes):
    try:
        st.image(img_bytes, use_container_width=True)
    except TypeError:
        st.image(img_bytes)

def _db_total_rows() -> int:
    path = os.environ["SHAJARA_DB_PATH"]
    try:
        with sqlite3.connect(path) as con:
            cur = con.execute("SELECT COUNT(*) FROM posts")
            return int(cur.fetchone()[0] or 0)
    except Exception:
        return 0

def _fetch_latest_for_channels(channels: List[str], limit_rows: int = 200) -> pd.DataFrame:
    if not channels:
        return pd.DataFrame()
    q_marks = ",".join("?" for _ in channels)
    sql = f"""SELECT id, datetime_utc, source_name, author, text, likes, shares, comments
              FROM posts
              WHERE source_name IN ({q_marks})
              ORDER BY datetime_utc DESC
              LIMIT ?"""
    with sqlite3.connect(os.environ["SHAJARA_DB_PATH"]) as con:
        df = pd.read_sql_query(sql, con, params=[*channels, limit_rows])
    return df

# --------- Non-blocking popup helper ---------
@contextmanager
def busy_popup(message: str = "Working…"):
    holder = st.empty()
    html = f"""
    <style>
      ._overlay {{
        position: fixed; inset: 0;
        background: rgba(0,0,0,.28);
        display: flex; align-items: center; justify-content: center;
        z-index: 999999999;
        pointer-events: none;   /* keep scrolling enabled */
      }}
      ._dialog {{
        pointer-events: none;
        background: #fff; border-radius: 12px; padding: 22px 26px; width: min(560px, 92vw);
        box-shadow: 0 12px 32px rgba(0,0,0,.25); text-align: center; font-family: Arial, sans-serif;
      }}
      ._spin {{
        width: 36px; height: 36px; border-radius: 50%;
        border: 4px solid #e5e7eb; border-top-color: #2563eb;
        animation: _sp 1s linear infinite; margin: 0 auto 10px auto;
      }}
      @keyframes _sp {{ to {{ transform: rotate(360deg); }} }}
      ._title {{ font-weight: 800; font-size: 18px; color: #111; margin-bottom: 4px; }}
      ._sub   {{ font-size: 12px; color: #555; }}
    </style>
    <div class="_overlay">
      <div class="_dialog">
        <div class="_spin"></div>
        <div class="_title">{message}</div>
        <div class="_sub">This closes automatically when finished.</div>
      </div>
    </div>
    """
    holder.markdown(html, unsafe_allow_html=True)
    try:
        yield
    finally:
        holder.empty()

# ---------- Sidebar ----------
st.sidebar.title("Settings")

# --- Collector controls (PHASE 1) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Collector options")

if "channels_initialized" not in st.session_state:
    st.session_state["channels_list"] = _load_initial_channels()
    st.session_state["queries_list"]  = _load_initial_search_terms()
    st.session_state["channels_initialized"] = True

channels_text = st.sidebar.text_area(
    "Channels (one per line)",
    value="\n".join(st.session_state["channels_list"]),
    height=140,
    key="channels_text_area",
    help="Examples: @SANANewsEnglish, https://t.me/syriageneral",
)
queries_text = st.sidebar.text_area(
    "Search queries (optional; one per line)",
    value="\n".join(st.session_state["queries_list"]),
    height=100,
    key="queries_text_area",
    help="e.g., السويداء\nSANA\nHama",
)
collect_limit_txt = st.sidebar.text_input("Per-channel fetch limit", value="1000")
collect_limit = _to_pos_int(collect_limit_txt, 1000)

current_channels = [ln.strip() for ln in channels_text.splitlines() if ln.strip()]
current_queries  = [ln.strip() for ln in queries_text.splitlines() if ln.strip()]

# Export to env for child
os.environ["TELEGRAM_CHANNELS"]     = json.dumps(current_channels, ensure_ascii=False)
os.environ["TELEGRAM_SEARCH_TERMS"] = json.dumps(current_queries,  ensure_ascii=False)
os.environ["TELEGRAM_LIMIT"]        = str(collect_limit)

c_ch1, c_ch2, c_ch3 = st.sidebar.columns(3)
if c_ch1.button("Save channels locally"):
    _save_channels_locally(current_channels)
    st.sidebar.success("Saved to .streamlit/channels.json")

if c_ch2.button("Reset channels from secrets"):
    default_from_secrets = _parse_list_from_str_or_json(S("TELEGRAM_CHANNELS", "") or "")
    st.session_state["channels_list"] = default_from_secrets
    st.session_state["channels_text_area"] = "\n".join(default_from_secrets)
    os.environ["TELEGRAM_CHANNELS"] = json.dumps(default_from_secrets, ensure_ascii=False)
    st.sidebar.info("Channels reset from secrets.toml")

if c_ch3.button("Clear channels & queries"):
    st.session_state["channels_list"] = []
    st.session_state["queries_list"]  = []
    st.session_state["channels_text_area"] = ""
    st.session_state["queries_text_area"]  = ""
    os.environ["TELEGRAM_CHANNELS"]     = "[]"
    os.environ["TELEGRAM_SEARCH_TERMS"] = "[]"

# --- Loader/analysis controls (PHASE 2) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Loader & analysis")

max_rows_txt = st.sidebar.text_input("Rows to load", value="10")
max_rows = _to_pos_int(max_rows_txt, 10)

search_txt = st.sidebar.text_input("Filter (LIKE)", value="")
since      = st.sidebar.text_input("Since (YYYY-MM-DD)", value="")
until      = st.sidebar.text_input("Until (YYYY-MM-DD)", value="")

limit_gpt_txt = st.sidebar.text_input("Analyze first N posts", value="10")
limit_gpt = _to_pos_int(limit_gpt_txt, 10)

drop_empty = st.sidebar.checkbox("Drop empty texts", value=True)
min_len_txt = st.sidebar.text_input("Minimum text length", value="1")
min_len = _to_pos_int(min_len_txt, 1)

# ---------- Actions ----------
colb1, colb2, colb3 = st.sidebar.columns(3)
run_click   = colb1.button("Run analysis")
tg_click    = colb2.button("Run Telegram collector")
fb_click    = colb3.button("Run Facebook collector")

if run_click:
    st.session_state["RUN_ANALYSIS_ONCE"] = True
run_analysis = st.session_state.pop("RUN_ANALYSIS_ONCE", False)

# ---------- Collector runner with live progress ----------
_channel_line = re.compile(r"Channel\s+(.+?)\s*->\s*fetched\s+(\d+)\s+messages", re.I)
_upsert_line1 = re.compile(r"Upserted\s+(\d+)\s+rows", re.I)
_upsert_line2 = re.compile(r"SQLite:\s*upserted\s+(\d+)\s+rows", re.I)

def _run_collector(module_path: str, label: str):
    env = os.environ.copy()
    env["SHAJARA_DB_PATH"]        = os.environ["SHAJARA_DB_PATH"]  # stays as 'shajara.db' unless env override
    env["TELEGRAM_CHANNELS"]      = os.environ.get("TELEGRAM_CHANNELS", "[]")
    env["TELEGRAM_SEARCH_TERMS"]  = os.environ.get("TELEGRAM_SEARCH_TERMS", "[]")
    env["TELEGRAM_LIMIT"]         = os.environ.get("TELEGRAM_LIMIT", "1000")

    cmd = [sys.executable, "-m", module_path]
    st.sidebar.info(f"Running: {' '.join(cmd)}")
    log_expander = st.sidebar.expander("Execution log", expanded=True)
    log_box = log_expander.empty()
    start = time.time(); lines: List[str] = []

    per_channel: Dict[str, int] = {}
    upserted_total = 0

    with busy_popup(f"Running {label}…"):
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
            )
        except Exception as e:
            st.sidebar.error(f"Failed to start: {e}")
            return

        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                lines.append(line)
                if len(lines) > 500: lines = lines[-500:]

                m = _channel_line.search(line)
                if m:
                    ch = m.group(1).strip()
                    try:
                        count = int(m.group(2))
                    except Exception:
                        count = 0
                    per_channel[ch] = count

                mu = _upsert_line1.search(line) or _upsert_line2.search(line)
                if mu:
                    try:
                        upserted_total = int(mu.group(1))
                    except Exception:
                        pass

                if per_channel:
                    live_table = pd.DataFrame(
                        [{"channel": k, "fetched": v} for k, v in sorted(per_channel.items())]
                    )
                    with log_expander:
                        st.dataframe(live_table, use_container_width=True, height=160)
                log_box.text("\n".join(lines))
        finally:
            code = proc.wait()
            dur = time.time() - start
            st.session_state["collect_result"] = {
                "ok": code == 0,
                "per_channel": per_channel,
                "upserted_total": upserted_total,
                "channels_used": current_channels,
                "limit": collect_limit,
                "elapsed_s": dur,
                "module": module_path,
            }
            if code == 0:
                st.sidebar.success(f"Completed in {dur:.1f}s (exit={code})")
            else:
                st.sidebar.error(f"Exited with code {code} after {dur:.1f}s")
            log_box.text("\n".join(lines))

# ---------- MAIN ----------
st.title("📡 SHAJARA — Analyze Telegram/Facebook Posts")

# PHASE 1: Collect
if tg_click:
    _run_collector("collectors.telegram_collector", "Telegram collector")

if fb_click:
    _run_collector("collectors.facebook_collector", "Facebook collector")

# If we just collected, show a summary + preview (before generic loader)
if "collect_result" in st.session_state:
    cr = st.session_state["collect_result"]
    st.header("✅ Collection summary & preview")
    cols = st.columns(4)
    cols[0].metric("Status", "OK" if cr["ok"] else "Error")
    cols[1].metric("Upserted rows (total)", cr.get("upserted_total", 0))
    cols[2].metric("Channels", len(cr.get("per_channel", {})))
    cols[3].metric("Per-channel limit", cr.get("limit", 0))

    if cr.get("per_channel"):
        st.dataframe(
            pd.DataFrame(
                [{"channel": k, "fetched": v} for k, v in sorted(cr["per_channel"].items())]
            ),
            use_container_width=True, height=180
        )

    with busy_popup("Building collection preview…"):
        preview_df = _fetch_latest_for_channels(cr.get("channels_used", []), limit_rows=min(200, max(50, cr.get("upserted_total", 200))))
    if not preview_df.empty:
        st.subheader("Latest rows from collected channels")
        st.dataframe(preview_df, use_container_width=True, height=300)
    else:
        st.info("No preview rows (maybe channels list was empty or collector returned no texts).")

# PHASE 2: Load from DB (independent of collection)
with busy_popup("Loading data…"):
    df = fetch_telegram_posts(
        limit=max_rows,
        since_utc=(since.strip() + "T00:00:00Z") if since.strip() else None,
        until_utc=(until.strip() + "T23:59:59Z") if until.strip() else None,
        search=search_txt.strip() or None,
    )

# Apply empty-text filtering
matched_before = len(df)
if not df.empty and "text" in df.columns and drop_empty:
    df = df[df["text"].astype(str).str.strip().str.len() >= min_len].copy()

total_rows = _db_total_rows()
st.info(
    f"Showing **{len(df)}** rows (after filters) out of **~{total_rows}** total. "
    f"Pulled latest **{max_rows}** rows; "
    f"LIKE={search_txt or '∅'}; "
    f"date range: {since or '∅'} → {until or '∅'}; "
    f"dropped-empty={'ON' if drop_empty else 'OFF'} (min_len={min_len})."
)

if df.empty:
    st.warning("No data matches your current filters. Increase 'Rows to load', clear LIKE, or disable 'Drop empty texts'.")
    st.stop()

st.success(f"Loaded {len(df)} rows from {os.environ['SHAJARA_DB_PATH']}")

# ---------- Stats ----------
st.header("🧮 Basic Statistics")
with busy_popup("Computing basic statistics…"):
    def _row_stats(row) -> Dict[str, Any]:
        s = token_stats((row.get("text") or ""), model=S("OPENAI_MODEL", OPENAI_MODEL))
        s["id"] = row.get("id")
        s["datetime_utc"] = row.get("datetime_utc")
        s["source"] = row.get("source_name")
        return s
    stat_df = pd.DataFrame(df.apply(_row_stats, axis=1).tolist())

c1,c2,c3,c4 = st.columns(4)
c1.metric("Posts", len(df))
c2.metric("Median (words)", int(stat_df["words"].median()) if not stat_df.empty else 0)
c3.metric("Median (chars)", int(stat_df["chars"].median()) if not stat_df.empty else 0)
c4.metric("Median (GPT tokens)", int(stat_df["gpt_tokens"].dropna().median()) if not stat_df.empty and stat_df["gpt_tokens"].notna().any() else 0)

with st.expander("Show table", expanded=True):
    st.dataframe(df[["datetime_utc","source_name","author","text","likes","shares","comments"]],
                 use_container_width=True, height=400)

with busy_popup("Rendering histogram…"):
    fig = plt.figure()
    stat_df["words"].plot(kind="hist", bins=30)
    plt.title("Word Count Distribution"); plt.xlabel("Words"); plt.ylabel("Frequency")
    st.pyplot(fig)

# ---------- Analysis ----------
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

st.header("🧠 Analysis (topics, entities, summary, sentiment)")
if run_analysis:
    with busy_popup("Analyzing posts & building visuals…"):
        if "gpt_cache" not in st.session_state:
            st.session_state["gpt_cache"] = {}

        local_cache = dict(st.session_state["gpt_cache"])
        cache_lock = Lock()

        results: List[Dict] = []
        progress = st.progress(0)
        N = min(int(limit_gpt), len(df))

        def worker(i: int, text: str) -> Dict[str, Any]:
            key = f"{S('OPENAI_MODEL', OPENAI_MODEL)}|" + hashlib.sha1((text or "").encode("utf-8")).hexdigest()
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
                progress.progress(int(done / N * 100))
        progress.empty()

        # Canonicalize
        all_topics_raw, all_entities_raw = [], []
        for r in results:
            all_topics_raw += extract_topics(r.get("topics"))
            all_entities_raw += extract_entities(r.get("entities"))

        uniq_topics   = sorted({t.strip() for t in all_topics_raw if t and t.strip()})
        uniq_entities = sorted({e.strip() for e in all_entities_raw if e and e.strip()})

        topic_map = gpt_semantic_merge_terms(uniq_topics) if uniq_topics else {}
        if not topic_map: topic_map = {k:k for k in uniq_topics}
        ent_map = gpt_semantic_merge_terms(uniq_entities) if uniq_entities else {}
        if not ent_map: ent_map = {k:k for k in uniq_entities}

        canon_results: List[Dict[str,Any]] = []
        for r in results:
            topics_en = [t.strip() for t in extract_topics(r.get("topics")) if t]
            entities_en = [e.strip() for e in extract_entities(r.get("entities")) if e]
            topics_en = list(dict.fromkeys([topic_map.get(t, t) for t in topics_en]))
            entities_en = list(dict.fromkeys([ent_map.get(e, e) for e in entities_en]))
            row = dict(r)
            row["topics"] = topics_en
            row["entities"] = entities_en
            canon_results.append(row)

        res_df = pd.DataFrame(canon_results)
        st.subheader("Raw LLM Results")
        st.dataframe(res_df, use_container_width=True, height=360)

        # Charts
        if "sentiment" in res_df.columns and res_df["sentiment"].notna().any():
            counts = res_df["sentiment"].value_counts(dropna=True)
            fig2 = plt.figure()
            counts.plot(kind="bar")
            plt.title("Sentiment Distribution"); plt.xlabel("Sentiment"); plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig2)

        all_topics_vals = []
        if "topics" in res_df.columns:
            for tlist in res_df["topics"].dropna():
                if isinstance(tlist, list):
                    all_topics_vals.extend([t for t in tlist if t])
        if all_topics_vals:
            topics_df = pd.DataFrame({"topic": all_topics_vals})
            top_topics = topics_df["topic"].value_counts().head(25)
            fig4 = plt.figure()
            top_topics.plot(kind="bar")
            plt.title("Top Topics"); plt.xlabel("Topic"); plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig4)

        # Graph + Wordcloud
        def build_graph_html():
            from modules.viz import build_cooccurrence_graph
            return build_cooccurrence_graph(canon_results, top_k_terms=50, min_edge_weight=1)

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
            from wordcloud import WordCloud as _WC
            wc = _WC(width=1300, height=550, background_color="white",
                     max_words=300, collocations=False, prefer_horizontal=1.0
            ).generate_from_frequencies(dict(items))
            buf = io.BytesIO(); wc.to_image().save(buf, format="PNG"); buf.seek(0)
            return buf

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_graph = ex.submit(build_graph_html)
            f_wc    = ex.submit(build_wordcloud_png)
            graph_html = f_graph.result()
            wc_buf     = f_wc.result()

        st.subheader("Co-occurrence Graph (PyVis)")
        st.components.v1.html(graph_html, height=700, scrolling=True)

        st.subheader("Word Cloud — Entities")
        if wc_buf is None:
            st.info("Not enough entities or 'wordcloud' package missing.")
        else:
            img = wc_buf.getvalue()
            _st_image(img)
            st.download_button("Download Word Cloud (PNG)", data=img, file_name="entities_wordcloud.png", mime="image/png")
else:
    st.info("Click **Run analysis** from the sidebar.")
