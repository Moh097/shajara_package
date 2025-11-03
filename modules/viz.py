# modules/viz.py
from __future__ import annotations
import json, re
from collections import Counter
from typing import Any, Dict, List, Tuple
from pyvis.network import Network

def _cooccurrence(topics_by_post: List[List[str]]) -> Tuple[Counter, Counter]:
    term_freq, edge_freq = Counter(), Counter()
    for terms in topics_by_post:
        if not terms:
            continue
        uniq = sorted(set(terms))
        term_freq.update(uniq)
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                edge_freq[(uniq[i], uniq[j])] += 1
    return term_freq, edge_freq

def build_cooccurrence_graph(
    results: List[Dict[str, Any]],
    top_k_terms: int = 25,
    min_edge_weight: int = 2,
    height: str = "680px",
    width: str = "100%",
    notebook: bool = False,
) -> str:
    """
    IMPORTANT: Do NOT reshape Arabic here. Let the browser handle shaping/bidi.
    We just pass plain Arabic labels and set a font face in vis options.
    """
    topics_by_post: List[List[str]] = []
    for r in results or []:
        topics = r.get("topics") or []
        if isinstance(topics, str):
            # tolerate both JSON and delimited strings
            try:
                import json as _json
                parsed = _json.loads(topics)
                if isinstance(parsed, list):
                    topics = [str(x) for x in parsed if x]
                else:
                    import re as _re
                    topics = [t.strip() for t in _re.split(r"[;,/|]+", topics) if t.strip()]
            except Exception:
                import re as _re
                topics = [t.strip() for t in _re.split(r"[;,/|]+", topics) if t.strip()]
        topics_by_post.append([str(t) for t in topics])

    term_freq, edge_freq = _cooccurrence(topics_by_post)
    if not term_freq:
        return "<div style='padding:1rem'>لا توجد مواضيع لبناء الرسم الشبكي.</div>"

    top_terms = [t for t, _ in term_freq.most_common(max(1, top_k_terms))]
    top_set = set(top_terms)

    net = Network(height=height, width=width, directed=False, notebook=notebook, cdn_resources="in_line")

    # Use an Arabic-capable font stack; browser will pick the first available.
    vis_options = {
        "locale": "ar",
        "interaction": {"hover": True, "tooltipDelay": 120},
        "nodes": {
            "shape": "dot",
            "scaling": {"min": 10, "max": 42},
            "font": {
                "size": 16,
                # Critical: use a CSS stack, not reshaped text.
                "face": "Tahoma, 'Noto Naskh Arabic', 'Noto Sans Arabic', 'Traditional Arabic', Arial, 'DejaVu Sans', sans-serif"
            },
        },
        "edges": {"smooth": {"type": "continuous"}, "selectionWidth": 2, "hoverWidth": 1.5},
        "physics": {
            "enabled": True,
            "stabilization": {"iterations": 200},
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {"gravitationalConstant": -50, "springLength": 120, "springConstant": 0.06},
        },
    }
    net.set_options(json.dumps(vis_options, ensure_ascii=False))

    max_freq = max((term_freq[t] for t in top_set), default=1)
    for t in top_terms:
        freq = term_freq[t]
        size = 10 + int(30 * (freq / max_freq))
        # No arabic_reshaper here — pass raw label
        net.add_node(n_id=t, label=t, title=f"{t} — {freq}", value=freq, size=size)

    for (a, b), w in edge_freq.items():
        if w < max(1, min_edge_weight):
            continue
        if a not in top_set or b not in top_set:
            continue
        net.add_edge(a, b, value=w, title=f"{w}", width=1 + (w - 1) * 0.8)

    if not net.edges:
        return "<div style='padding:1rem'>لا توجد حواف بعد الفلترة. خفّض الحد الأدنى.</div>"

    return net.generate_html(notebook=notebook)
