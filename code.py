import json
import time
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st

# Optional Mermaid renderer (fallback to showing code if not available)
try:
    from streamlit_mermaid import st_mermaid
    MERMAID_OK = True
except Exception:
    MERMAID_OK = False

st.set_page_config(page_title="Pocket IT Architect Assistant", layout="wide")

# -----------------------------
# Sample Embedded Data (editable)
# -----------------------------
# Architecture patterns dictionary: lightweight scoring across common concerns.
PATTERNS = [
    {
        "name": "Microservices",
        "good_for": ["independent deploy", "scale teams", "evolutionary", "polyglot"],
        "concerns": {"availability": 4, "latency": 3, "consistency": 3, "throughput": 4, "cost": 3, "complexity": 4},
        "pros": ["Independent deployment", "Team autonomy", "Fine-grained scalability"],
        "cons": ["Operational complexity", "Distributed transactions", "Observability overhead"]
    },
    {
        "name": "Event-Driven (Pub/Sub)",
        "good_for": ["real-time", "async", "decoupling", "streaming", "event sourcing"],
        "concerns": {"availability": 4, "latency": 4, "consistency": 2, "throughput": 5, "cost": 3, "complexity": 3},
        "pros": ["Loose coupling", "High throughput", "Scalable consumers"],
        "cons": ["Eventual consistency", "Ordering challenges", "Schema evolution management"]
    },
    {
        "name": "Layered (N-Tier)",
        "good_for": ["simplicity", "web apps", "enterprise standard", "monolith->modular"],
        "concerns": {"availability": 3, "latency": 3, "consistency": 4, "throughput": 3, "cost": 3, "complexity": 2},
        "pros": ["Simple mental model", "Easy to start", "Clear separation of concerns"],
        "cons": ["Can become rigid", "Scaling needs care", "Layer coupling"]
    },
    {
        "name": "Serverless",
        "good_for": ["spiky traffic", "pay-per-use", "rapid prototyping", "managed operations"],
        "concerns": {"availability": 4, "latency": 3, "consistency": 3, "throughput": 3, "cost": 4, "complexity": 2},
        "pros": ["No server ops", "Auto-scale", "Cost-effective at low/medium load"],
        "cons": ["Cold starts", "Vendor lock-in", "Execution time limits"]
    },
    {
        "name": "Data Lakehouse",
        "good_for": ["analytics", "batch+stream", "ML", "BI"],
        "concerns": {"availability": 3, "latency": 2, "consistency": 3, "throughput": 5, "cost": 3, "complexity": 3},
        "pros": ["Open formats", "Unified lake+warehouse", "Scales for analytics"],
        "cons": ["Ops maturity varies", "Governance needs rigor", "Query performance tuning"]
    }
]

# Technology dataset (very small & editable)
TECH = [
    {"category": "Messaging", "name": "Apache Kafka", "throughput": 5, "latency": 3, "durability": 5, "ops_maturity": 4, "cost": 3, "cloud_fit": 4, "notes": "Great for streams, high throughput"},
    {"category": "Messaging", "name": "RabbitMQ", "throughput": 3, "latency": 4, "durability": 4, "ops_maturity": 5, "cost": 3, "cloud_fit": 4, "notes": "Great for work queues, request/reply"},
    {"category": "Messaging", "name": "Azure Service Bus", "throughput": 3, "latency": 3, "durability": 4, "ops_maturity": 4, "cost": 3, "cloud_fit": 5, "notes": "Managed messaging on Azure"},
    {"category": "DB", "name": "PostgreSQL", "throughput": 4, "latency": 4, "durability": 5, "ops_maturity": 5, "cost": 4, "cloud_fit": 5, "notes": "Relational, strong ecosystem"},
    {"category": "DB", "name": "MongoDB", "throughput": 4, "latency": 3, "durability": 4, "ops_maturity": 4, "cost": 3, "cloud_fit": 5, "notes": "Document model, flexible schema"},
    {"category": "API Gateway", "name": "Kong", "throughput": 4, "latency": 4, "durability": 3, "ops_maturity": 4, "cost": 4, "cloud_fit": 4, "notes": "Open-source, plugins"},
    {"category": "API Gateway", "name": "Azure API Management", "throughput": 4, "latency": 3, "durability": 4, "ops_maturity": 5, "cost": 3, "cloud_fit": 5, "notes": "Managed, great governance"}
]

# Lightweight compliance checklist (subset)
CHECKLIST = [
    {"domain": "Security", "control": "AuthN/AuthZ defined (RBAC/OAuth2)", "criticality": "High"},
    {"domain": "Security", "control": "Secrets managed (KMS/Vault)", "criticality": "High"},
    {"domain": "Resilience", "control": "RTO/RPO documented", "criticality": "High"},
    {"domain": "Data", "control": "PII classification & retention", "criticality": "High"},
    {"domain": "Ops", "control": "Logging/Tracing (OpenTelemetry)", "criticality": "Medium"},
    {"domain": "Ops", "control": "Automated CI/CD with quality gates", "criticality": "Medium"},
    {"domain": "Compliance", "control": "Region/data residency considered", "criticality": "Medium"},
]

REQ_TOKENS = {
    "availability": ["ha", "multi-region", "disaster recovery", "availability", "uptime", "rto", "rpo"],
    "latency": ["low latency", "real-time", "sub-second", "interactive", "fast"],
    "consistency": ["strong consistency", "acid", "transactions", "ordering"],
    "throughput": ["high throughput", "batch", "streaming", "events", "iot"],
    "cost": ["cost", "budget", "pay per use", "optimize", "opex", "capex"],
    "complexity": ["simple", "operate", "ops", "observability", "team size"]
}

# -----------------------------
# Helpers
# -----------------------------
def score_patterns(requirements_text: str, weights: Dict[str, int]) -> pd.DataFrame:
    text = requirements_text.lower()
    # Basic token hit scoring for each concern
    hits = {k: sum(1 for t in v if t in text) for k, v in REQ_TOKENS.items()}
    # Normalize hits into [0..5] roughly
    max_hit = max(hits.values()) if hits else 1
    norm = {k: (v / max_hit) * 5 if max_hit > 0 else 0 for k, v in hits.items()}

    rows = []
    for p in PATTERNS:
        # Weighted score: user emphasis * pattern inherent capability
        s = 0
        details = {}
        for c in ["availability", "latency", "consistency", "throughput", "cost", "complexity"]:
            val = weights.get(c, 3) * ((p["concerns"].get(c, 3) + norm.get(c, 0)) / 2.0)
            details[c] = round(val, 2)
            s += val
        rows.append({
            "Pattern": p["name"],
            "Score": round(s, 2),
            "Pros": ", ".join(p["pros"]),
            "Cons": ", ".join(p["cons"]),
            **details
        })
    df = pd.DataFrame(rows).sort_values("Score", ascending=False)
    return df

def tech_df() -> pd.DataFrame:
    return pd.DataFrame(TECH)

def compute_tech_scores(df: pd.DataFrame, weights: Dict[str, int]) -> pd.DataFrame:
    cols = ["throughput", "latency", "durability", "ops_maturity", "cloud_fit"]
    # Cost is inverse (lower cost = better)
    df = df.copy()
    df["cost_inv"] = 6 - df["cost"]
    total = 0
    for c in cols:
        df[f"{c}_w"] = df[c] * weights.get(c, 3)
        total += df[f"{c}_w"]
    total += df["cost_inv"] * weights.get("cost", 3)
    df["WeightedScore"] = total
    return df.sort_values(["category", "WeightedScore"], ascending=[True, False])

def make_mermaid(pattern: str, picks: Dict[str, str]) -> str:
    """
    Produce a simple Mermaid diagram from selections.
    """
    cloud = picks.get("cloud", "Cloud")
    msg = picks.get("messaging", "Messaging")
    db = picks.get("database", "Database")
    api = picks.get("apigw", "API Gateway")
    client = picks.get("client", "Client App")
    core = f"{pattern} Core"

    return f"""
flowchart LR
  subgraph {cloud}
    API[{api}] --> Svc[{core}]
    Svc --> MQ[{msg}]
    Svc --> DB[({db})]
  end
  Client[{client}] --> API
"""

def save_exports(pattern_row: pd.Series, tech_choices: Dict[str, str], checklist_status: pd.DataFrame, notes: str):
    export = {
        "timestamp": int(time.time()),
        "recommended_pattern": pattern_row.get("Pattern", ""),
        "pattern_score": float(pattern_row.get("Score", 0)) if "Score" in pattern_row else 0,
        "tech_choices": tech_choices,
        "checklist": checklist_status.to_dict(orient="records"),
        "notes": notes,
    }
    # Ensure folder exists
    import os
    os.makedirs("exports", exist_ok=True)
    # JSON
    with open("exports/decisions.json", "w") as f:
        json.dump(export, f, indent=2)
    # CSV (flatten simple parts)
    flat_rows = []
    for k, v in tech_choices.items():
        flat_rows.append({"key": k, "value": v})
    pd.DataFrame(flat_rows).to_csv("exports/decisions.csv", index=False)
    return export

# -----------------------------
# UI
# -----------------------------
st.title("🧭 Pocket IT Architect Assistant")
st.caption("Quick decision support for IT Architects: patterns, tech trade-offs, compliance, and diagrams.")

tabs = st.tabs([
    "1) Use Case → Pattern",
    "2) Tech Comparator",
    "3) Compliance",
    "4) Diagram",
    "5) Notes & Export"
])

# -----------------------------
# Tab 1: Pattern Recommender
# -----------------------------
with tabs[0]:
    st.subheader("Describe your use case")
    colA, colB = st.columns([2, 1])
    with colA:
        req_text = st.text_area(
            "Business/technical requirements (free text)",
            height=180,
            placeholder="e.g., Real-time order tracking for IoT devices with high availability and low latency. Budget-sensitive."
        )
    with colB:
        st.markdown("**Emphasis (1–5)**")
        weights = {}
        for w in ["availability", "latency", "consistency", "throughput", "cost", "complexity"]:
            weights[w] = st.slider(w.capitalize(), 1, 5, 3)

    if st.button("Recommend Patterns", type="primary"):
        dfp = score_patterns(req_text, weights)
        st.success("Top recommendations")
        st.dataframe(dfp, use_container_width=True)
        if len(dfp):
            st.session_state["top_pattern"] = dfp.iloc[0].to_dict()
        else:
            st.session_state["top_pattern"] = {}

# -----------------------------
# Tab 2: Tech Comparator
# -----------------------------
with tabs[1]:
    st.subheader("Compare technologies")
    df = tech_df()

    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Category", sorted(df["category"].unique().tolist()))
    with col2:
        st.markdown("**Weights (1–5)**")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        w_tp = st.slider("Throughput", 1, 5, 4)
        w_lat = st.slider("Latency", 1, 5, 4)
    with cc2:
        w_dur = st.slider("Durability", 1, 5, 4)
        w_ops = st.slider("Ops Maturity", 1, 5, 4)
    with cc3:
        w_cf = st.slider("Cloud Fit", 1, 5, 4)
        w_cost = st.slider("Cost (lower is better)", 1, 5, 4)

    subset = df[df["category"] == category]
    ranked = compute_tech_scores(subset, {
        "throughput": w_tp, "latency": w_lat, "durability": w_dur,
        "ops_maturity": w_ops, "cloud_fit": w_cf, "cost": w_cost
    })
    st.dataframe(ranked[["name", "WeightedScore", "throughput", "latency", "durability", "ops_maturity", "cloud_fit", "cost", "notes"]],
                 use_container_width=True)

    st.markdown("**Pick your choice for each category**")
    picks = {
        "messaging": st.selectbox("Messaging", df[df["category"] == "Messaging"]["name"].tolist(), index=0),
        "database": st.selectbox("Database", df[df["category"] == "DB"]["name"].tolist(), index=0),
        "apigw": st.selectbox("API Gateway", df[df["category"] == "API Gateway"]["name"].tolist(), index=0),
        "client": st.text_input("Client (e.g., Mobile, Web SPA)", value="Web SPA"),
        "cloud": st.text_input("Cloud/Platform label", value="Azure"),
    }
    st.session_state["tech_picks"] = picks

# -----------------------------
# Tab 3: Compliance
# -----------------------------
with tabs[2]:
    st.subheader("Compliance & security checks")
    cl_df = pd.DataFrame(CHECKLIST)
    status = []
    for i, row in cl_df.iterrows():
        done = st.checkbox(f'[{row["domain"]}] {row["control"]} ({row["criticality"]})', value=False, key=f"chk_{i}")
        status.append({"domain": row["domain"], "control": row["control"], "criticality": row["criticality"], "done": done})
    status_df = pd.DataFrame(status)
    progress = (status_df["done"].mean() if len(status_df) else 0.0) * 100
    st.progress(progress/100.0, text=f"Completion: {progress:.0f}%")
    st.session_state["checklist_status"] = status_df

# -----------------------------
# Tab 4: Diagram
# -----------------------------
with tabs[3]:
    st.subheader("Auto-diagram (Mermaid)")
    pattern = (st.session_state.get("top_pattern", {}) or {}).get("Pattern", "Layered (N-Tier)")
    picks = st.session_state.get("tech_picks", {}) or {}
    mm = make_mermaid(pattern, picks)
    st.code(mm, language="mermaid")
    if MERMAID_OK:
        st.info("Rendering Mermaid (via streamlit-mermaid).")
        st_mermaid(mm)
    else:
        st.warning("To render the diagram, install the optional component: `pip install streamlit-mermaid`")

# -----------------------------
# Tab 5: Notes & Export
# -----------------------------
with tabs[4]:
    st.subheader("Decision notes")
    notes = st.text_area("Rationale, risks, assumptions, and decisions (RRAD)", height=160,
                         placeholder="Why this pattern? Trade-offs? Security implications? Migration plan?")

    if st.button("Export decisions (CSV + JSON)"):
        top = st.session_state.get("top_pattern", {}) or {}
        picks = st.session_state.get("tech_picks", {}) or {}
        status = st.session_state.get("checklist_status", pd.DataFrame())
        export = save_exports(pd.Series(top), picks, status, notes)
        st.success("Exported to /exports/decisions.json and /exports/decisions.csv")
        st.json(export)
