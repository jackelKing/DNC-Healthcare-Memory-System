"""Streamlit UI: DNC-Based Personal Causal Memory System for Healthcare."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, logging
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

from data.patients import PATIENTS, get_patient_by_id, get_patient_names
from core.patient_session import get_or_create_session
from explainability.explainer import build_explanation, explanation_to_json
from ollama_client.client import is_ollama_running, list_models

logging.basicConfig(level=logging.INFO)
st.set_page_config(
    page_title="Causal Memory - Healthcare DNC",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
h1,h2,h3{font-family:'IBM Plex Mono',monospace;}
.stApp{background:#0d1117;color:#c9d1d9;}
.mcard{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px 18px;margin-bottom:10px;}
.cstep{background:#161b22;border-left:3px solid #58a6ff;padding:9px 14px;margin:5px 0;border-radius:0 6px 6px 0;font-family:'IBM Plex Mono',monospace;font-size:12px;color:#8b949e;}
.cbar{height:5px;background:linear-gradient(90deg,#1d4ed8,#3b82f6);border-radius:3px;margin-top:4px;}
.ok{color:#3fb950;}.warn{color:#d29922;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## DNC Healthcare")
    st.markdown("---")
    ollama_ok = is_ollama_running()
    status_label = "Ollama Online" if ollama_ok else "Ollama Offline (fallback)"
    status_cls = "ok" if ollama_ok else "warn"
    st.markdown(f'<span class="{status_cls}">{status_label}</span>', unsafe_allow_html=True)
    if ollama_ok:
        models = list_models()
        if models:
            st.caption("Models: " + ", ".join(models[:4]))
    st.markdown("---")
    patient_names = get_patient_names()
    opts = {f"{pid} -- {name}": pid for pid, name in patient_names.items()}
    sel = st.selectbox("Patient", list(opts.keys()))
    pid = opts[sel]
    patient = get_patient_by_id(pid)
    st.markdown(
        f'<div class="mcard"><b>{patient["name"]}</b><br>'
        f'Age {patient["age"]} | {patient["sex"]} | {len(patient["events"])} events</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    top_k = st.slider("Top-K results", 1, 8, 4)
    use_llm = st.toggle("LLM explanation", value=ollama_ok)
    do_filter = st.checkbox("Year range filter")
    y_start = y_end = None
    if do_filter:
        c1, c2 = st.columns(2)
        y_start = c1.number_input("From", 2019, 2025, 2019)
        y_end   = c2.number_input("To",   2019, 2025, 2024)
    st.markdown("---")
    st.markdown("**Example queries**")
    examples = [
        "What caused the kidney disease?",
        "Why did the patient develop neuropathy?",
        "Downstream effects of diabetes?",
        "What triggered the heart failure?",
        "Why did the patient need an ICD?",
        "What caused the cerebral thrombosis?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["qi"] = ex

# ── Main tabs ─────────────────────────────────────────────────────────────────
st.markdown("# DNC Causal Memory -- Healthcare")
st.markdown("*Differentiable Neural Computer-style patient reasoning*")
st.markdown("---")
tabs = st.tabs([
    "Query and Reasoning",
    "Timeline",
    "Causal Graph",
    "Memory Matrix",
    "Debug",
])

# Tab 1 -- Query ──────────────────────────────────────────────────────────────
with tabs[0]:
    query = st.text_input(
        "Clinical question:",
        value=st.session_state.get("qi", ""),
        placeholder="e.g. What caused the kidney disease?",
    )
    if st.button("Run Causal Query", type="primary", use_container_width=True) and query.strip():
        with st.spinner("Querying DNC memory and traversing causal graph..."):
            session = get_or_create_session(patient)
            result = session.query(
                query.strip(),
                year_start=int(y_start) if y_start else None,
                year_end=int(y_end) if y_end else None,
                top_k=top_k,
            )
            explanation = build_explanation(result, use_llm=use_llm)
            st.session_state["res"] = result
            st.session_state["exp"] = explanation

    if "res" in st.session_state:
        result = st.session_state["res"]
        explanation = st.session_state["exp"]

        st.markdown("### Explanation")
        st.info(explanation.get("llm_explanation") or explanation.get("rule_explanation", ""))
        st.markdown("---")

        ca, cb = st.columns(2)
        with ca:
            st.markdown("### Matched Events")
            for ev in result.matched_events:
                conf = result.confidence_scores.get(ev.get("id", ""), 0.0)
                pct = int(conf * 100)
                st.markdown(
                    f'<div class="mcard">'
                    f'<b>{ev.get("condition","")}</b> '
                    f'<span style="color:#8b949e">({ev.get("timestamp","")})</span><br>'
                    f'<small>Slot #{ev.get("_slot","?")} | Confidence: {conf:.1%}</small>'
                    f'<div class="cbar" style="width:{pct}%"></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        with cb:
            st.markdown("### Ranked Causes")
            for i, c in enumerate(result.ranked_causes[:6], 1):
                sc = c.get("causal_score", 0)
                pct2 = int(sc * 100)
                st.markdown(
                    f'<div class="mcard">'
                    f'<b>{i}. {c.get("condition","")}</b> '
                    f'<span style="color:#8b949e;font-size:11px">(hop {c.get("hop","?")})</span><br>'
                    f'<small style="color:#6e7681">{c.get("mechanism","")}</small><br>'
                    f'<small>Score: {sc:.4f}</small>'
                    f'<div class="cbar" style="width:{pct2}%;background:linear-gradient(90deg,#7928ca,#ff0080)"></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("### Reasoning Chain")
        for step in result.reasoning_chain:
            st.markdown(f'<div class="cstep">{step}</div>', unsafe_allow_html=True)

        if result.causal_traversal_path:
            path_str = " -> ".join(result.causal_traversal_path)
            st.markdown(f"**Traversal path:** `{path_str}`")

# Tab 2 -- Timeline ───────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### Patient Event Timeline")
    events = patient["events"]
    fig = go.Figure()
    for i, ev in enumerate(events):
        yp = i % 3
        alpha = 0.4 + ev.get("severity", 0.5) * 0.6
        col = f"rgba(88,166,255,{alpha:.2f})"
        fig.add_trace(go.Scatter(
            x=[ev["timestamp"]], y=[yp],
            mode="markers+text",
            marker=dict(
                size=18 + ev.get("severity", 0.5) * 20,
                color=col,
                line=dict(width=2, color="#58a6ff"),
            ),
            text=[ev["condition"]],
            textposition="top center",
            name=ev["id"],
            showlegend=False,
            hovertemplate=(
                f"<b>{ev['condition']}</b><br>"
                f"Year: {ev['timestamp']}<br>"
                f"Severity: {ev.get('severity',0):.0%}"
                "<extra></extra>"
            ),
        ))
        for lnk in ev.get("causal_links", []):
            ce = next((e for e in events if e["id"] == lnk["cause"]), None)
            if ce:
                fig.add_annotation(
                    x=ev["timestamp"], y=yp,
                    ax=ce["timestamp"], ay=events.index(ce) % 3,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowcolor="#f85149",
                    arrowsize=1.2, arrowwidth=1.5, opacity=0.6,
                )
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9"),
        xaxis=dict(title="Year", gridcolor="#21262d"),
        yaxis=dict(visible=False),
        height=340, margin=dict(t=20, b=40, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    df = pd.DataFrame([{
        "ID": e["id"], "Year": e["timestamp"], "Condition": e["condition"],
        "Severity": e.get("severity", 0.5), "Causal Links": len(e.get("causal_links", [])),
    } for e in events])
    st.dataframe(df, use_container_width=True, hide_index=True)

# Tab 3 -- Causal Graph ───────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### Causal Relationship Graph")
    session = get_or_create_session(patient)
    session.load()
    G = session.causal_graph.graph
    pos = nx.spring_layout(G, seed=42, k=2.5)
    ex_x, ex_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ex_x += [x0, x1, None]
        ex_y += [y0, y1, None]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=ex_x, y=ex_y, mode="lines",
        line=dict(width=1.5, color="#f85149"),
        opacity=0.5, hoverinfo="none", showlegend=False,
    ))
    fig2.add_trace(go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        mode="markers+text",
        marker=dict(
            size=[12 + G.nodes[n].get("severity", 0.5) * 22 for n in G.nodes()],
            color=[G.nodes[n].get("severity", 0.5) for n in G.nodes()],
            colorscale="Blues", showscale=True,
            line=dict(width=2, color="#58a6ff"),
        ),
        text=[G.nodes[n].get("condition", n)[:20] for n in G.nodes()],
        textposition="top center",
        hovertext=[
            f"<b>{G.nodes[n].get('condition','')}</b><br>"
            f"Year: {G.nodes[n].get('timestamp','')} | "
            f"Sev: {G.nodes[n].get('severity',0):.0%}"
            for n in G.nodes()
        ],
        hoverinfo="text", showlegend=False,
    ))
    fig2.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9"),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=500, margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig2, use_container_width=True)
    edge_df = pd.DataFrame([
        {"Source": u, "Target": v, "Mechanism": d.get("mechanism", "")}
        for u, v, d in G.edges(data=True)
    ])
    if not edge_df.empty:
        st.dataframe(edge_df, use_container_width=True, hide_index=True)

# Tab 4 -- Memory Matrix ──────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### DNC Memory Matrix Heatmap")
    session = get_or_create_session(patient)
    session.load()
    active = [i for i, s in enumerate(session.controller.memory.slots) if s is not None]
    if active:
        sub = session.controller.memory.matrix_snapshot()[active, :80]
        labels = []
        for i in active:
            sl = session.controller.memory.slots[i]
            ev = session.controller.event_index.get(sl.event_id, {}) if sl else {}
            labels.append(f"Slot {i}: {ev.get('condition','?')[:24]}")
        fig3 = go.Figure(go.Heatmap(
            z=sub, y=labels, colorscale="Blues",
            colorbar=dict(title="Activation", thickness=12, bgcolor="#161b22"),
        ))
        fig3.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9"),
            xaxis=dict(title="Dimension (0-79)", gridcolor="#21262d"),
            height=80 + len(active) * 28,
            margin=dict(t=10, b=40, l=220, r=20),
        )
        st.plotly_chart(fig3, use_container_width=True)

        usage_vals = [session.controller.memory.usage[i] for i in active]
        fig4 = go.Figure(go.Bar(x=labels, y=usage_vals, marker_color="#58a6ff", opacity=0.8))
        fig4.update_layout(
            title="Slot Usage Scores",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9"),
            xaxis=dict(tickangle=-35, gridcolor="#21262d"),
            yaxis=dict(title="Usage", gridcolor="#21262d"),
            height=300, margin=dict(t=30, b=130, l=40, r=20),
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Run a query first to populate the memory matrix.")

# Tab 5 -- Debug ──────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### Debug Panel")
    if "res" in st.session_state and "exp" in st.session_state:
        result = st.session_state["res"]
        explanation = st.session_state["exp"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Read Weights (top 10)")
            cw = result.read_weights_content
            top10 = np.argsort(cw)[::-1][:10]
            session = get_or_create_session(patient)
            df_w = pd.DataFrame([{
                "Slot": int(i),
                "Content": round(float(cw[i]), 6),
                "Forward": round(float(result.read_weights_forward[i]), 6),
                "Backward": round(float(result.read_weights_backward[i]), 6),
                "Event": (
                    session.controller.memory.slots[i].event_id
                    if session.controller.memory.slots[i] else "empty"
                ),
            } for i in top10])
            st.dataframe(df_w, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("#### Cosine Similarities")
            from ollama_client.client import embed_text
            q_vec = embed_text(result.query)
            sims = []
            for i, sl in enumerate(session.controller.memory.slots):
                if sl:
                    mv = session.controller.memory.matrix[i]
                    denom = np.linalg.norm(q_vec) * np.linalg.norm(mv) + 1e-9
                    sim = float(np.dot(q_vec, mv) / denom)
                    ev = session.controller.event_index.get(sl.event_id, {})
                    sims.append({
                        "Condition": ev.get("condition", ""),
                        "Cosine": round(sim, 4),
                        "Slot": i,
                    })
            sd = pd.DataFrame(sorted(sims, key=lambda x: x["Cosine"], reverse=True)[:8])
            if not sd.empty:
                st.dataframe(sd, use_container_width=True, hide_index=True)
        st.markdown("#### Full JSON")
        st.code(explanation_to_json(explanation), language="json")
    else:
        st.info("Run a query in the Query tab first.")

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#484f58;font-size:11px;'
    'font-family:IBM Plex Mono,monospace">'
    "DNC Causal Memory | Healthcare AI | Streamlit + Ollama"
    "</p>",
    unsafe_allow_html=True,
)