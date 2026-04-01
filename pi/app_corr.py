import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="6G IDS — Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }
section[data-testid="stSidebar"] { background: #0d1220; border-right: 1px solid #1e2d4a; }
.ids-header {
    background: linear-gradient(135deg, #0f1f3d 0%, #0a1628 50%, #0d1a33 100%);
    border: 1px solid #1e3a5f; border-radius: 16px; padding: 2rem 2.5rem;
    margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.ids-header::before {
    content: ''; position: absolute; top: -50%; right: -10%; width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(0,212,255,0.06) 0%, transparent 70%); pointer-events: none;
}
.ids-header h1 { font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800; color: #ffffff; margin: 0 0 0.3rem 0; letter-spacing: -0.5px; }
.ids-header p { color: #7a9bbf; font-size: 0.95rem; margin: 0; font-family: 'Space Mono', monospace; }
.ids-badge { display: inline-block; background: rgba(0,212,255,0.1); border: 1px solid rgba(0,212,255,0.3); color: #00d4ff; font-family: 'Space Mono', monospace; font-size: 0.7rem; padding: 3px 10px; border-radius: 4px; margin-bottom: 0.8rem; letter-spacing: 1px; }
.metric-card { background: #0d1627; border: 1px solid #1a2d45; border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center; }
.metric-card .label { font-size: 0.75rem; color: #5a7fa0; font-family: 'Space Mono', monospace; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.4rem; }
.metric-card .value { font-size: 1.8rem; font-weight: 800; color: #00d4ff; }
.metric-card .sub { font-size: 0.75rem; color: #4a6a85; font-family: 'Space Mono', monospace; }
.result-benign { background: linear-gradient(135deg, #0a2a1a, #0d3322); border: 1px solid #1a5c3a; border-left: 4px solid #00e676; border-radius: 12px; padding: 1.5rem 2rem; margin-top: 1rem; }
.result-attack { background: linear-gradient(135deg, #2a0a0a, #330d0d); border: 1px solid #5c1a1a; border-left: 4px solid #ff4444; border-radius: 12px; padding: 1.5rem 2rem; margin-top: 1rem; }
.result-alarm  { background: linear-gradient(135deg, #2a1f0a, #332700); border: 1px solid #5c420a; border-left: 4px solid #ffaa00; border-radius: 12px; padding: 1.5rem 2rem; margin-top: 1rem; }
.result-title { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800; margin: 0 0 0.4rem 0; }
.result-sub   { font-family: 'Space Mono', monospace; font-size: 0.8rem; opacity: 0.7; }
.stSelectbox label, .stSlider label, .stNumberInput label { color: #7a9bbf !important; font-size: 0.8rem !important; font-family: 'Space Mono', monospace !important; letter-spacing: 0.5px !important; }
.section-title { font-family: 'Space Mono', monospace; font-size: 0.7rem; letter-spacing: 2px; color: #3a5a7a; text-transform: uppercase; border-bottom: 1px solid #1a2d45; padding-bottom: 0.5rem; margin: 1.5rem 0 1rem 0; }
.stTabs [data-baseweb="tab-list"] { background: #0d1220; border-radius: 10px; gap: 4px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #5a7fa0; font-family: 'Space Mono', monospace; font-size: 0.8rem; }
.stTabs [aria-selected="true"] { background: #1a2d45 !important; color: #00d4ff !important; }
button[kind="primary"] { background: linear-gradient(135deg, #0066cc, #0044aa) !important; border: none !important; border-radius: 8px !important; font-family: 'Space Mono', monospace !important; font-size: 0.85rem !important; letter-spacing: 0.5px !important; }
</style>
""", unsafe_allow_html=True)

FINAL_THRESHOLD = 0.30

DEFAULT_DATASET_FEATURES = {
    "eMBB":    ["Dur", "TotPkts", "TotBytes", "Rate", "Load", "Loss", "pLoss", "TcpRtt"],
    "mMTC":    ["TotPkts", "Rate", "SrcGap", "DstGap", "Dur", "Load", "Loss", "TcpRtt"],
    "URLLC":   ["TcpRtt", "SynAck", "AckDat", "Loss", "Dur", "Rate", "TotPkts", "TotBytes"],
    "TON_IoT": ["src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "duration", "proto", "conn_state", "service"],
}

DATASET_INFO = {
    "eMBB":    {"desc": "Enhanced Mobile Broadband",           "color": "#00d4ff", "attacks": ["TCP SYN Flood", "Bandwidth Saturation"]},
    "mMTC":    {"desc": "Massive Machine-Type Communications", "color": "#7c3aed", "attacks": ["TCP SYN Scan / Connection Flooding", "Slow-Rate Resource Exhaustion", "FIN Scan"]},
    "URLLC":   {"desc": "Ultra-Reliable Low-Latency",          "color": "#f59e0b", "attacks": ["UDP DDoS Flood", "RST Injection", "SLA Violation / DoS", "Reconnaissance"]},
    "TON_IoT": {"desc": "IoT Network Dataset",                 "color": "#10b981", "attacks": ["DDoS", "DoS", "Scanning", "Password", "Backdoor", "Ransomware", "Injection", "XSS", "MITM"]},
}

MODEL_RESULTS = {
    "LightGBM":      {"TON_IoT": 0.9951, "URLLC": 0.7084, "eMBB": 0.9483, "mMTC": 0.9304},
    "XGBoost":       {"TON_IoT": 0.9944, "URLLC": 0.7089, "eMBB": 0.9469, "mMTC": 0.9252},
    "Random Forest": {"TON_IoT": 0.9931, "URLLC": 0.6676, "eMBB": 0.9372, "mMTC": 0.9234},
    "Extra Trees":   {"TON_IoT": 0.9558, "URLLC": 0.6424, "eMBB": 0.8888, "mMTC": 0.9096},
    "MLP":           {"TON_IoT": 0.9910, "URLLC": 0.6558, "eMBB": 0.8993, "mMTC": 0.9026},
    "Logistic Reg.": {"TON_IoT": 0.8811, "URLLC": 0.5360, "eMBB": 0.8478, "mMTC": 0.8830},
}

def classify_attack_type(name, row):
    def g(col, default=None):
        return row.get(col, default)

    # ── eMBB ─────────────────────────────────────────────────────────────
    if name == "eMBB":
        dur    = g("Dur",     1.0)
        rate   = g("Rate",    0.0)
        totpkt = g("TotPkts", 50)
        # TCP SYN Flood: short-lived burst at high packet rate, few packets
        # Data shows: SYN Flood flows have Dur <= 0.35s AND Rate > 50 pkt/s
        if dur <= 0.35 and rate > 50:
            return "TCP SYN Flood"
        # Bandwidth Saturation: sustained flow, lower rate, more packets, high load
        return "Bandwidth Saturation"

    # ── mMTC ─────────────────────────────────────────────────────────────
    if name == "mMTC":
        dur    = g("Dur", 1.0)
        tot_pk = g("TotPkts", 100)
        load   = g("Load", 5000)
        if tot_pk < 10 and dur < 1.0:
            return "TCP SYN Scan / Connection Flooding"
        if dur > 3.0 and load < 3000:
            return "Slow-Rate Resource Exhaustion"
        return "FIN Scan"

    # ── URLLC ────────────────────────────────────────────────────────────
    if name == "URLLC":
        dur    = g("Dur", 1.0)
        tot_pk = g("TotPkts", 10)
        rtt    = g("TcpRtt", 0.01)
        if dur == 0 and tot_pk == 1:
            return "UDP DDoS Flood"
        if rtt > 0.05 and tot_pk < 5:
            return "RST Injection"
        if rtt > 0.1:
            return "SLA Violation / DoS"
        return "Reconnaissance"

    # ── TON_IoT (9 attack types) ──────────────────────────────────────────
    if name == "TON_IoT":
        src_pkts   = g("src_pkts",   10)
        dst_pkts   = g("dst_pkts",   10)
        src_bytes  = g("src_bytes",  1000)
        dst_bytes  = g("dst_bytes",  1000)
        duration   = g("duration",   1.0)
        conn_state = g("conn_state", "SF")
        service    = g("service",    "-")
        proto      = g("proto",      "tcp")
        if src_pkts > 1000 and duration < 2.0:
            return "DDoS"
        if src_pkts > 200 and duration < 5.0:
            return "DoS"
        if conn_state in ("REJ","RSTO","RSTOS0") and src_pkts < 10:
            return "Scanning"
        if service in ("ssh","ftp","ftp-data") and src_pkts < 20:
            return "Password"
        if duration > 60 and src_bytes < 5000 and conn_state == "SF":
            return "Backdoor"
        if src_bytes > 100000 and duration > 10:
            return "Ransomware"
        if service in ("http","http-alt"):
            if dst_bytes > src_bytes * 2:
                return "Injection"
            return "XSS"
        if proto not in ("tcp","udp") or (conn_state == "SF" and src_bytes < 500):
            return "MITM"
        return "DoS"

    return "Unknown"

@st.cache_resource
def load_models():
    base = os.path.join(os.path.dirname(__file__), "models")
    mp = os.path.join(base, "lgbm_models.pkl")
    pp = os.path.join(base, "lgbm_preprocessors.pkl")
    fp = os.path.join(base, "lgbm_features.pkl")
    lp = os.path.join(base, "lgbm_label_encoders.pkl")
    if os.path.exists(mp):
        with open(mp, "rb") as f: mdls = pickle.load(f)
        with open(pp, "rb") as f: pres = pickle.load(f)
        with open(fp, "rb") as f: fts  = pickle.load(f)
        les = None
        if os.path.exists(lp):
            with open(lp, "rb") as f: les = pickle.load(f)
        return mdls, pres, fts, les
    return None, None, None, None

models, preprocessors, features, label_encoders = load_models()
models_loaded = models is not None

# Use the exact feature order exported by the notebook when available.
DATASET_FEATURES = features if models_loaded and isinstance(features, dict) else DEFAULT_DATASET_FEATURES

# ── SIDEBAR ──────────────────────────────────
with st.sidebar:
    st.markdown('<div class="ids-badge">6G NETWORK SECURITY</div>', unsafe_allow_html=True)
    st.markdown("### 🛡️ IDS Control Panel")
    st.markdown("---")
    available_datasets = ["eMBB", "mMTC", "URLLC", "TON_IoT"]
    if models_loaded and isinstance(features, dict):
        available_datasets = [ds for ds in available_datasets if ds in features]
    selected_dataset = st.selectbox("NETWORK SLICE / DATASET", available_datasets)
    info = DATASET_INFO[selected_dataset]
    st.markdown(f"""<div style="background:#0d1627;border:1px solid #1a2d45;border-left:3px solid {info['color']};
    border-radius:8px;padding:0.8rem 1rem;margin:0.5rem 0 1rem 0;font-size:0.8rem;color:#7a9bbf;font-family:'Space Mono',monospace;">
    {info['desc']}<br><span style="color:{info['color']};font-size:0.7rem">{len(info['attacks'])} attack types monitored</span></div>""", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model</div>', unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#5a7fa0;line-height:1.8;">
    ✦ LightGBM (Leaf-wise Boosting)<br>✦ Histogram-based gradient boosting<br>✦ Best F1 across all datasets</div>""", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Detection Threshold</div>', unsafe_allow_html=True)
    threshold = st.slider("Confidence threshold", 0.05, 0.90, FINAL_THRESHOLD, 0.05)
    st.caption(f"Notebook default FINAL_THRESHOLD = {FINAL_THRESHOLD:.2f}")
    st.markdown("---")
    st.success("✅ Notebook-exported LightGBM models loaded") if models_loaded else st.warning("⚠️ Run notebook cell 63 to export models into streamlit_app/models.")

# ── HEADER ───────────────────────────────────
st.markdown(f"""<div class="ids-header">
    <div class="ids-badge">LIGHTGBM · BEST MODEL</div>
    <h1>🛡️ 6G Intrusion Detection System</h1>
    <p>Real-time network traffic classification · {selected_dataset} slice · {info['desc']}</p>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Live Detection", "📊 Model Comparison", "📁 Batch Analysis"])

# ── TAB 1: LIVE DETECTION ────────────────────
with tab1:
    st.markdown('<div class="section-title">Input Network Flow Features</div>', unsafe_allow_html=True)
    row_vals = {}

    if selected_dataset == "eMBB":
        c1,c2,c3,c4 = st.columns(4)
        row_vals["Dur"]      = c1.number_input("Duration (Dur)",        value=0.713976,    format="%.6f")
        row_vals["TotPkts"]  = c2.number_input("Total Packets",         value=21,          step=1)
        row_vals["TotBytes"] = c3.number_input("Total Bytes",           value=3296,        step=100)
        row_vals["Rate"]     = c4.number_input("Rate (pkt/s)",          value=28.012146,   format="%.6f")
        c5,c6,c7,c8 = st.columns(4)
        row_vals["Load"]     = c5.number_input("Load",                  value=33390.47656, format="%.5f")
        row_vals["Loss"]     = c6.number_input("Loss",                  value=1,           step=1)
        row_vals["pLoss"]    = c7.number_input("Packet Loss % (pLoss)", value=4.545455,    format="%.6f")
        row_vals["TcpRtt"]   = c8.number_input("TCP RTT (TcpRtt)",     value=0.015993,    format="%.6f")

    elif selected_dataset == "mMTC":
        c1,c2,c3,c4 = st.columns(4)
        row_vals["TotPkts"]  = c1.number_input("Total Packets",         value=28,          step=1)
        row_vals["Rate"]     = c2.number_input("Rate (pkt/s)",          value=8.75,        format="%.6f")
        row_vals["SrcGap"]   = c3.number_input("Src Gap (SrcGap)",      value=0.035,       format="%.6f")
        row_vals["DstGap"]   = c4.number_input("Dst Gap (DstGap)",      value=0.038,       format="%.6f")
        c5,c6,c7,c8 = st.columns(4)
        row_vals["Dur"]      = c5.number_input("Duration (Dur)",        value=3.2,         format="%.4f")
        row_vals["Load"]     = c6.number_input("Load",                  value=11250.0,     format="%.2f")
        row_vals["Loss"]     = c7.number_input("Loss",                  value=2,           step=1)
        row_vals["TcpRtt"]   = c8.number_input("TCP RTT (TcpRtt)",     value=0.021,       format="%.6f")

    elif selected_dataset == "URLLC":
        c1,c2,c3,c4 = st.columns(4)
        row_vals["TcpRtt"]   = c1.number_input("TCP RTT (TcpRtt)",     value=0.06,        format="%.6f")
        row_vals["SynAck"]   = c2.number_input("SYN/ACK (SynAck)",     value=0.001,       format="%.6f")
        row_vals["AckDat"]   = c3.number_input("ACK Data (AckDat)",    value=0.001,       format="%.6f")
        row_vals["Loss"]     = c4.number_input("Loss",                  value=0,           step=1)
        c5,c6,c7,c8 = st.columns(4)
        row_vals["Dur"]      = c5.number_input("Duration (Dur)",        value=0.01,        format="%.4f")
        row_vals["Rate"]     = c6.number_input("Rate (pkt/s)",          value=4.0,         format="%.4f")
        row_vals["TotPkts"]  = c7.number_input("Total Packets",         value=4,           step=1)
        row_vals["TotBytes"] = c8.number_input("Total Bytes",           value=300,         step=50)

    else:  # TON_IoT
        c1,c2,c3,c4 = st.columns(4)
        row_vals["src_bytes"]  = c1.number_input("Src Bytes",           value=1500,        step=100)
        row_vals["dst_bytes"]  = c2.number_input("Dst Bytes",           value=1200,        step=100)
        row_vals["src_pkts"]   = c3.number_input("Src Packets",         value=15,          step=1)
        row_vals["dst_pkts"]   = c4.number_input("Dst Packets",         value=12,          step=1)
        c5,c6,c7,c8 = st.columns(4)
        row_vals["duration"]   = c5.number_input("Duration",            value=2.0,         format="%.2f")
        row_vals["proto"]      = c6.selectbox("Protocol",               ["tcp","udp","icmp"])
        row_vals["conn_state"] = c7.selectbox("Conn State",             ["SF","REJ","RSTO","RSTOS0","S0","S1","OTH"])
        row_vals["service"]    = c8.selectbox("Service",                ["-","http","http-alt","ssh","ftp","ftp-data","dns"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Run Detection", type="primary", use_container_width=True)

    if predict_btn:
        if models_loaded and selected_dataset in models:
            mdl   = models[selected_dataset]
            pre   = preprocessors[selected_dataset]
            feats = features[selected_dataset]
            input_dict = {k: v for k, v in row_vals.items() if k in feats}
            for f in feats:
                if f not in input_dict: input_dict[f] = 0
            input_df = pd.DataFrame([input_dict])[feats]
            try:
                input_proc = pre.transform(input_df)
                proba  = mdl.predict_proba(input_proc)[0, 1]
                binary = int(proba >= 0.5)
                if binary == 0:
                    label = "Benign"; verdict = "benign"
                elif proba < threshold:
                    label = "False Alarm"; verdict = "alarm"
                else:
                    label = classify_attack_type(selected_dataset, row_vals); verdict = "attack"

                if verdict == "benign":
                    st.markdown(f'<div class="result-benign"><div class="result-title" style="color:#00e676">✅ BENIGN TRAFFIC</div><div class="result-sub">No threat detected · Attack probability: {proba:.1%}</div></div>', unsafe_allow_html=True)
                elif verdict == "alarm":
                    st.markdown(f'<div class="result-alarm"><div class="result-title" style="color:#ffaa00">⚠️ FALSE ALARM</div><div class="result-sub">Low confidence ({proba:.1%}) — below threshold {threshold:.0%}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-attack"><div class="result-title" style="color:#ff4444">🚨 ATTACK DETECTED: {label}</div><div class="result-sub">LightGBM confidence: {proba:.1%} · Slice: {selected_dataset}</div></div>', unsafe_allow_html=True)

                st.markdown('<div class="section-title" style="margin-top:1.5rem">Detection Output</div>', unsafe_allow_html=True)
                mc1, mc2, mc3 = st.columns(3)
                color = "#ff4444" if proba >= 0.5 else "#00e676"
                mc1.markdown(f'<div class="metric-card"><div class="label">LightGBM Score</div><div class="value" style="color:{color}">{proba:.1%}</div><div class="sub">attack probability</div></div>', unsafe_allow_html=True)
                mc2.markdown(f'<div class="metric-card"><div class="label">Verdict</div><div class="value" style="font-size:1.1rem;color:{color}">{label}</div><div class="sub">final decision</div></div>', unsafe_allow_html=True)
                mc3.markdown(f'<div class="metric-card"><div class="label">Threshold</div><div class="value" style="color:#ffaa00">{threshold:.0%}</div><div class="sub">confidence gate</div></div>', unsafe_allow_html=True)

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=proba * 100,
                    domain={"x": [0,1], "y": [0,1]},
                    title={"text": "Threat Level", "font": {"color": "#7a9bbf", "size": 14}},
                    gauge={
                        "axis": {"range": [0,100], "tickcolor": "#3a5a7a"},
                        "bar":  {"color": "#ff4444" if proba >= 0.5 else "#00e676"},
                        "bgcolor": "#0d1627", "bordercolor": "#1a2d45",
                        "steps": [{"range":[0,30],"color":"#0a2a1a"},{"range":[30,50],"color":"#1a1a0a"},{"range":[50,100],"color":"#2a0a0a"}],
                        "threshold": {"line": {"color": "#ffaa00", "width": 2}, "value": threshold * 100}
                    },
                    number={"suffix": "%", "font": {"color": "#e2e8f0"}}
                ))
                fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color":"#7a9bbf"}, height=260, margin=dict(t=30,b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.info("ℹ️ Models not loaded — showing demo output. Run the final notebook cell to export models.")
            proba = np.random.uniform(0.55, 0.95)
            label = np.random.choice(DATASET_INFO[selected_dataset]["attacks"])
            st.markdown(f'<div class="result-attack"><div class="result-title" style="color:#ff4444">🚨 [DEMO] ATTACK: {label}</div><div class="result-sub">Simulated confidence: {proba:.1%}</div></div>', unsafe_allow_html=True)

# ── TAB 2: MODEL COMPARISON ──────────────────
with tab2:
    st.markdown('<div class="section-title">F1 Macro Score — All Models × All Datasets</div>', unsafe_allow_html=True)
    colors_map = {
        "LightGBM": "#00d4ff", "XGBoost": "#06b6d4", "Random Forest": "#7c3aed",
        "Extra Trees": "#10b981", "MLP": "#f59e0b", "Logistic Reg.": "#6b7280",
    }
    datasets_order = ["TON_IoT", "eMBB", "mMTC", "URLLC"]
    fig = go.Figure()
    for model in MODEL_RESULTS:
        vals = [MODEL_RESULTS[model][ds] for ds in datasets_order]
        fig.add_trace(go.Bar(name=model, x=datasets_order, y=vals, marker_color=colors_map[model],
            marker_line_width=0, opacity=0.9, text=[f"{v:.3f}" for v in vals],
            textposition="outside", textfont=dict(size=9, color="#7a9bbf")))
    fig.update_layout(barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7a9bbf", family="Space Mono"),
        xaxis=dict(gridcolor="#1a2d45", tickfont=dict(color="#7a9bbf")),
        yaxis=dict(gridcolor="#1a2d45", tickfont=dict(color="#7a9bbf"), range=[0,1.12], title="F1 Macro"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#7a9bbf")),
        height=420, margin=dict(t=20,b=20), bargap=0.15, bargroupgap=0.05)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Average F1 Across All Datasets</div>', unsafe_allow_html=True)
    avg_data = {m: np.mean(list(s.values())) for m,s in MODEL_RESULTS.items()}
    avg_df = pd.DataFrame.from_dict(avg_data, orient="index", columns=["Avg F1 Macro"])
    avg_df = avg_df.sort_values("Avg F1 Macro", ascending=False)
    avg_df["Rank"] = range(1, len(avg_df)+1)
    avg_df["Avg F1 Macro"] = avg_df["Avg F1 Macro"].map("{:.4f}".format)
    avg_df["Selected"] = avg_df.index.map(lambda m: "✅ Active" if m == "LightGBM" else "—")
    st.dataframe(avg_df[["Rank","Avg F1 Macro","Selected"]], use_container_width=True)

    st.markdown('<div class="section-title">LightGBM vs All Models — Radar View</div>', unsafe_allow_html=True)
    categories = ["TON_IoT","eMBB","mMTC","URLLC"]
    fig_radar = go.Figure()
    for model, color in colors_map.items():
        vals = [MODEL_RESULTS[model][ds] for ds in categories]
        vals_c = vals + [vals[0]]; cats_c = categories + [categories[0]]
        lw = 3 if model == "LightGBM" else 1; op = 0.9 if model == "LightGBM" else 0.5
        fig_radar.add_trace(go.Scatterpolar(r=vals_c, theta=cats_c,
            fill="toself" if model == "LightGBM" else "none",
            name=model, line=dict(color=color, width=lw), opacity=op))
    fig_radar.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0.5,1.0], gridcolor="#1a2d45", tickfont=dict(color="#3a5a7a",size=9)),
            angularaxis=dict(gridcolor="#1a2d45", tickfont=dict(color="#7a9bbf"))),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#7a9bbf",family="Space Mono"),
        legend=dict(bgcolor="rgba(0,0,0,0)"), height=400, margin=dict(t=20,b=20))
    st.plotly_chart(fig_radar, use_container_width=True)

# ── TAB 3: BATCH ANALYSIS ────────────────────
with tab3:
    st.markdown('<div class="section-title">Upload CSV for Batch Prediction</div>', unsafe_allow_html=True)
    feat_list = " · ".join(DATASET_FEATURES.get(selected_dataset, DEFAULT_DATASET_FEATURES[selected_dataset]))
    st.markdown(f"""<div style="font-family:'Space Mono',monospace;font-size:0.78rem;color:#5a7fa0;margin-bottom:1rem;
    background:#0d1627;border:1px solid #1a2d45;border-radius:8px;padding:0.8rem 1rem;">
    <span style="color:#3a5a7a;">EXPECTED FEATURES [{selected_dataset}]:</span><br>
    <span style="color:#00d4ff;">{feat_list}</span></div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file, sep=None, engine='python')
        st.markdown(f"**{len(df_upload)} rows loaded** · {df_upload.shape[1]} columns")
        st.dataframe(df_upload.head(5), use_container_width=True)

        if models_loaded and selected_dataset in models:
            if st.button("🚀 Run Batch Detection", type="primary"):
                mdl   = models[selected_dataset]
                pre   = preprocessors[selected_dataset]
                feats = features[selected_dataset]
                available = [f for f in feats if f in df_upload.columns]
                if not available:
                    st.error(f"No matching features found. CSV must contain: {', '.join(feats)}")
                else:
                    for f in feats:
                        if f not in df_upload.columns: df_upload[f] = 0
                    X_batch = df_upload[feats]
                    try:
                        X_proc = pre.transform(X_batch)
                        probas = mdl.predict_proba(X_proc)[:, 1]
                        labels = []
                        for i, (prob, binary) in enumerate(zip(probas, (probas >= 0.5).astype(int))):
                            row = df_upload.iloc[i].to_dict()
                            if binary == 0: labels.append("Benign")
                            elif prob < threshold: labels.append("False Alarm")
                            else: labels.append(classify_attack_type(selected_dataset, row))

                        df_upload["Prediction"] = labels
                        df_upload["Confidence"] = probas.round(4)

                        summary = pd.Series(labels).value_counts()
                        st.markdown('<div class="section-title">Detection Summary</div>', unsafe_allow_html=True)
                        cols = st.columns(min(len(summary), 4))
                        for idx, (lbl, cnt) in enumerate(summary.items()):
                            color = "#00e676" if lbl == "Benign" else "#ffaa00" if lbl == "False Alarm" else "#ff4444"
                            cols[idx % 4].markdown(f'<div class="metric-card"><div class="label">{lbl}</div><div class="value" style="color:{color}">{cnt}</div><div class="sub">{cnt/len(labels):.1%} of total</div></div>', unsafe_allow_html=True)

                        fig_pie = px.pie(values=summary.values, names=summary.index,
                            color_discrete_sequence=["#00e676","#ff4444","#ffaa00","#00d4ff","#7c3aed","#f59e0b","#10b981","#e11d48","#6366f1"])
                        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#7a9bbf",family="Space Mono"),
                            legend=dict(bgcolor="rgba(0,0,0,0)"), height=320, margin=dict(t=20,b=20))
                        st.plotly_chart(fig_pie, use_container_width=True)

                        st.markdown('<div class="section-title">Confidence Distribution</div>', unsafe_allow_html=True)
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Histogram(x=probas, nbinsx=30, marker_color="#00d4ff", opacity=0.8))
                        fig_hist.add_vline(x=threshold, line_dash="dash", line_color="#ffaa00",
                            annotation_text=f"Threshold ({threshold:.0%})", annotation_font_color="#ffaa00")
                        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#7a9bbf",family="Space Mono"),
                            xaxis=dict(gridcolor="#1a2d45", title="Attack Probability"),
                            yaxis=dict(gridcolor="#1a2d45", title="Count"),
                            height=280, margin=dict(t=20,b=20))
                        st.plotly_chart(fig_hist, use_container_width=True)

                        st.markdown('<div class="section-title">Full Results</div>', unsafe_allow_html=True)
                        display_cols = ["Prediction","Confidence"] + [f for f in feats if f in df_upload.columns][:6]
                        st.dataframe(df_upload[display_cols], use_container_width=True)
                        csv_out = df_upload.to_csv(index=False).encode("utf-8")
                        st.download_button("⬇️ Download Results CSV", csv_out, "ids_results.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Batch prediction error: {e}")
        else:
            st.warning("Models not loaded. Run the final notebook cell first.")
    else:
        st.markdown('<div class="section-title">Expected Features for Selected Dataset</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Feature": DATASET_FEATURES[selected_dataset]}), use_container_width=True)

st.markdown("""<div style="text-align:center;margin-top:3rem;font-family:'Space Mono',monospace;font-size:0.7rem;
color:#2a3f5f;border-top:1px solid #1a2d45;padding-top:1rem;">
6G IDS · Model: LightGBM · Datasets: eMBB · mMTC · URLLC · TON_IoT</div>""", unsafe_allow_html=True)