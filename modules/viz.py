from __future__ import annotations
import json
import itertools
from typing import List, Dict, Any
from collections import Counter, defaultdict

import networkx as nx
from pyvis.network import Network


def _norm_token(x: Any) -> str:
    try:
        s = str(x).strip().lower()
    except Exception:
        s = ""
    return s


def _collect_terms(rows: List[Dict[str, Any]]) -> Counter:
    """Count terms once per row (topics + entities)."""
    freq = Counter()
    for r in rows:
        topics = r.get("topics") or []
        entities = r.get("entities") or []
        pool = set()
        for t in topics:
            tok = _norm_token(t)
            if tok:
                pool.add(tok)
        for e in entities:
            tok = _norm_token(e)
            if tok:
                pool.add(tok)
        for tok in pool:
            freq[tok] += 1
    return freq


def _cooccur_edges(rows: List[Dict[str, Any]], allowed: set[str]) -> Counter:
    """Build co-occurrence counts for unordered pairs within each row."""
    edges = Counter()
    for r in rows:
        topics = r.get("topics") or []
        entities = r.get("entities") or []
        pool = []
        for t in topics:
            tok = _norm_token(t)
            if tok and tok in allowed:
                pool.append(tok)
        for e in entities:
            tok = _norm_token(e)
            if tok and tok in allowed:
                pool.append(tok)
        pool = sorted(set(pool))  # unique per row
        for a, b in itertools.combinations(pool, 2):
            if a > b:
                a, b = b, a
            edges[(a, b)] += 1
    return edges


def _pyvis_html(net: Network) -> str:
    """Return rendered HTML for a PyVis Network."""
    if hasattr(net, "generate_html"):
        return net.generate_html()
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        path = tmp.name
    try:
        net.write_html(path)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def build_cooccurrence_graph(
    rows: List[Dict[str, Any]],
    top_k_terms: int = 50,
    min_edge_weight: int = 2,
    keep_isolates: bool = True,
    auto_relax_when_empty: bool = True,
    # PyVis/label styling
    font_size: int = 20,          # <— bigger labels
    bold_labels: bool = True,     # <— bold labels
    physics: bool = True,
    height: str = "650px",
    width: str = "100%",
    notebook: bool = False,
) -> str:
    """
    Build a Co-occurrence network HTML (PyVis) with bigger, bold node labels by default.
    """
    try:
        top_k_terms = max(1, int(top_k_terms))
    except Exception:
        top_k_terms = 50
    try:
        min_edge_weight = max(1, int(min_edge_weight))
    except Exception:
        min_edge_weight = 2

    # 1) term frequencies (unique per row)
    freq = _collect_terms(rows)
    if not freq:
        return "<div style='padding:12px;font:14px Arial'>No terms to show.</div>"

    # 2) select top-K terms
    top_terms = {w for w, _ in freq.most_common(top_k_terms)} or set(freq.keys())

    # 3) edges among top terms
    edges = _cooccur_edges(rows, top_terms)

    # 4) apply threshold; auto-relax once if nothing survives
    thr = min_edge_weight
    filtered_edges = [(a, b, w) for (a, b), w in edges.items() if w >= thr]
    if not filtered_edges and auto_relax_when_empty and edges:
        thr = 1
        filtered_edges = [(a, b, w) for (a, b), w in edges.items() if w >= 1]

    # 5) build the graph
    G = nx.Graph()
    for term in top_terms:
        G.add_node(term, size=max(10, 10 + 2 * int(freq[term])), freq=int(freq[term]))
    for a, b, w in filtered_edges:
        if a in G and b in G:
            G.add_edge(a, b, weight=int(w))

    if not keep_isolates and G.number_of_edges() > 0:
        isolates = [n for n, d in G.degree() if d == 0]
        G.remove_nodes_from(isolates)

    if G.number_of_nodes() == 0:
        return (
            "<div style='padding:12px;font:14px Arial'>"
            "Not enough co-occurrences to draw a graph. "
            "Increase <b>Top K terms</b> or lower <b>Min edge weight</b>."
            "</div>"
        )

    # 6) PyVis rendering
    net = Network(height=height, width=width, directed=False, notebook=notebook)
    net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=120, spring_strength=0.01, damping=0.9)
    net.toggle_physics(physics)

    # Add nodes with bigger/bold labels (use HTML <b> with font.multi='html')
    for n, data in G.nodes(data=True):
        label = f"<b>{n}</b>" if bold_labels else n
        net.add_node(
            n,
            label=label,
            title=f"{n} — freq={data.get('freq', 0)}",
            value=data.get("freq", 1),
        )

    for u, v, data in G.edges(data=True):
        w = int(data.get("weight", 1))
        net.add_edge(u, v, value=w, title=f"co-occurrence={w}")

    # Vis-network options (inject font size + enable HTML multi + bold style)
    net.set_options(f"""
    {{
      "nodes": {{
        "font": {{
          "size": {int(font_size)},
          "color": "#111",
          "face": "arial",
          "multi": "html",
          "bold": {{"size": {int(font_size)}, "face": "arial", "mod": "bold", "color": "#111"}}
        }},
        "scaling": {{"min": 10, "max": 70}}
      }},
      "edges": {{
        "color": {{"color": "#9ab"}},
        "smooth": false
      }},
      "physics": {{
        "enabled": true,
        "barnesHut": {{
          "gravitationalConstant": -20000,
          "centralGravity": 0.3,
          "springLength": 120,
          "springConstant": 0.01,
          "avoidOverlap": 0.2
        }},
        "timestep": 0.5,
        "stabilization": {{"iterations": 100}}
      }},
      "interaction": {{
        "hover": true,
        "tooltipDelay": 100
      }}
    }}
    """)

    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    info_html = (
        f"<div class='legend' style='padding:8px 12px; font:12px Arial; color:#555;'>"
        f"nodes={node_count}, edges={edge_count}, min_edge_weight={thr}, top_k_terms={len(top_terms)}"
        f"</div>"
    )

    html = _pyvis_html(net)
    if "</body>" in html:
        html = html.replace("</body>", info_html + "</body>")
    else:
        html += info_html
    return html
