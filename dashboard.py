import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, time

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Lender's Club | Risk Monitor",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="expanded"
)

# =============================================
# RAZORPAY-STYLE CSS
# =============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ===== GLOBAL ===== */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #0a0e27 !important; }
#MainMenu, footer, header { visibility: hidden; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0a0e27; }
::-webkit-scrollbar-thumb { background: #2563eb; border-radius: 4px; }

/* ===== SIDEBAR — Razorpay style ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060b1f 0%, #0d1333 100%) !important;
    border-right: 1px solid rgba(37, 99, 235, 0.15);
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* ===== KPI METRIC CARDS ===== */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #111640 0%, #0d1230 100%);
    border: 1px solid rgba(37, 99, 235, 0.12);
    border-radius: 14px;
    padding: 22px 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: all 0.35s cubic-bezier(.4,0,.2,1);
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(37, 99, 235, 0.4);
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(37, 99, 235, 0.12);
}
div[data-testid="stMetric"] label {
    color: #7c8db5 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px;
    text-transform: uppercase;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #e8edf5 !important;
    font-weight: 800 !important;
    font-size: 1.6rem !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] svg { display: none; }

/* ===== TABS — Razorpay pill style ===== */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1230;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(37, 99, 235, 0.1);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #7c8db5 !important;
    font-weight: 600;
    font-size: 0.82rem;
    padding: 8px 18px;
    letter-spacing: 0.3px;
}
.stTabs [aria-selected="true"] {
    background: #2563eb !important;
    color: #ffffff !important;
}

/* ===== CUSTOM COMPONENTS ===== */
.brand-bar {
    background: linear-gradient(135deg, #060b1f, #111845);
    border-bottom: 1px solid rgba(37, 99, 235, 0.2);
    padding: 20px 24px;
    margin: -1rem -1rem 20px -1rem;
    display: flex;
    align-items: center;
    gap: 14px;
}
.brand-logo {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; color: white; font-weight: 800;
}
.brand-text { color: #e8edf5; font-size: 1rem; font-weight: 700; letter-spacing: -0.3px; }
.brand-sub { color: #4a5a80; font-size: 0.68rem; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; }

.nav-section { color: #4a5a80; font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin: 24px 0 10px 0; padding: 0 4px; }

.stat-row {
    display: flex; gap: 8px; padding: 10px 12px;
    border-radius: 8px; margin: 4px 0;
    background: rgba(37, 99, 235, 0.04);
    border: 1px solid rgba(37, 99, 235, 0.06);
}
.stat-label { color: #7c8db5; font-size: 0.72rem; font-weight: 500; flex: 1; }
.stat-value { color: #e8edf5; font-size: 0.78rem; font-weight: 700; }

.page-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 28px; padding-bottom: 20px;
    border-bottom: 1px solid rgba(37, 99, 235, 0.1);
}
.page-title { color: #e8edf5; font-size: 1.5rem; font-weight: 800; letter-spacing: -0.5px; }
.page-badge {
    background: rgba(37, 99, 235, 0.15);
    color: #60a5fa; font-size: 0.7rem; font-weight: 600;
    padding: 5px 14px; border-radius: 20px;
    letter-spacing: 0.5px;
    display: flex; align-items: center; gap: 6px;
}
.live-dot {
    width: 7px; height: 7px; background: #22c55e; border-radius: 50%;
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(34,197,94,0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(34,197,94,0); }
}

.card {
    background: linear-gradient(145deg, #111640 0%, #0d1230 100%);
    border: 1px solid rgba(37, 99, 235, 0.1);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
}
.card-title {
    color: #e8edf5; font-size: 0.95rem; font-weight: 700;
    margin-bottom: 4px; display: flex; align-items: center; gap: 8px;
}
.card-desc { color: #5a6a8a; font-size: 0.75rem; font-weight: 400; margin-bottom: 16px; }

.risk-pill {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.5px;
}
.risk-low { background: rgba(34,197,94,0.15); color: #22c55e; }
.risk-med { background: rgba(234,179,8,0.15); color: #eab308; }
.risk-high { background: rgba(239,68,68,0.15); color: #ef4444; }

.info-banner {
    background: rgba(37, 99, 235, 0.06);
    border: 1px solid rgba(37, 99, 235, 0.12);
    border-radius: 10px;
    padding: 14px 18px;
    color: #8fa3c4; font-size: 0.8rem; line-height: 1.7;
    margin-bottom: 20px;
}

.footer-bar {
    text-align: center; padding: 30px 0 10px;
    border-top: 1px solid rgba(37, 99, 235, 0.08);
    margin-top: 40px;
}
.footer-text { color: #3a4a6a; font-size: 0.68rem; font-weight: 500; letter-spacing: 0.5px; }
</style>
""", unsafe_allow_html=True)

# =============================================
# PLOTLY THEME
# =============================================
PLOTLY_BG = "rgba(0,0,0,0)"
PLOTLY_LAYOUT = dict(
    paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
    font=dict(family="Inter", color="#8fa3c4", size=12),
    margin=dict(l=30, r=20, t=40, b=30),
    xaxis=dict(gridcolor="rgba(37,99,235,0.06)", zerolinecolor="rgba(37,99,235,0.08)"),
    yaxis=dict(gridcolor="rgba(37,99,235,0.06)", zerolinecolor="rgba(37,99,235,0.08)"),
    hoverlabel=dict(bgcolor="#111640", font_size=12, font_family="Inter", bordercolor="#2563eb"),
)
RISK_COLORS = {"Low": "#22c55e", "Medium": "#eab308", "High": "#ef4444"}
ACCENT = "#2563eb"
ACCENT2 = "#7c3aed"

# =============================================
# DATA LOAD
# =============================================
@st.cache_data
def load_data():
    fp = os.path.join(os.path.dirname(__file__), "final_dataset.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp)
    return pd.DataFrame()

df = load_data()
if df.empty:
    st.error("Dataset not found. Run the notebook first to generate final_dataset.csv")
    st.stop()

FRIENDLY = {
    "grade": "Loan Grade", "int_rate": "Interest Rate", "all_util": "Credit Utilization",
    "max_bal_bc": "Max Card Balance", "mths_since_rcnt_il": "Months Since Last Installment",
    "total_bal_il": "Total Installment Balance", "il_util": "Installment Utilization",
    "target": "Default", "prob": "Risk Score", "risk_bucket": "Risk Category"
}

# =============================================
# SIDEBAR — Razorpay style navigation
# =============================================
with st.sidebar:
    # Brand header
    st.markdown("""
    <div class="brand-bar">
        <div class="brand-logo">LC</div>
        <div>
            <div class="brand-text">Lender's Club</div>
            <div class="brand-sub">Risk Monitor</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "Go to",
        ["Overview", "Risk Analysis", "Stress Testing", "Feature Intelligence", "Data Explorer"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="nav-section">Filters</div>', unsafe_allow_html=True)

    risk_opts = sorted(df["risk_bucket"].dropna().unique().tolist())
    sel_risk = st.multiselect("Risk Category", risk_opts, default=risk_opts)

    prob_min, prob_max = float(df["prob"].min()), float(df["prob"].max())
    prob_range = st.slider("Risk Score Range", prob_min, prob_max, (prob_min, prob_max), format="%.2f")

    st.markdown('<div class="nav-section">Stress Config</div>', unsafe_allow_html=True)
    stress = st.slider("Stress Multiplier", 1.0, 2.5, 1.0, 0.1)
    lgd = st.slider("LGD %", 10, 100, 60, 5) / 100
    ead = st.number_input("Avg Exposure (Rs)", value=100000, step=10000)

    # Quick stats
    st.markdown('<div class="nav-section">Quick Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-row"><span class="stat-label">Total Records</span><span class="stat-value">{len(df):,}</span></div>
    <div class="stat-row"><span class="stat-label">Features</span><span class="stat-value">{len(df.columns)}</span></div>
    <div class="stat-row"><span class="stat-label">Default Rate</span><span class="stat-value">{df['target'].mean()*100:.1f}%</span></div>
    <div class="stat-row"><span class="stat-label">Avg Risk Score</span><span class="stat-value">{df['prob'].mean()*100:.1f}%</span></div>
    """, unsafe_allow_html=True)

# =============================================
# FILTER DATA
# =============================================
fdf = df[
    (df["risk_bucket"].isin(sel_risk)) &
    (df["prob"] >= prob_range[0]) &
    (df["prob"] <= prob_range[1])
].copy()
fdf["stressed_prob"] = (fdf["prob"] * stress).clip(upper=1.0)
fdf["EL"] = fdf["prob"] * lgd * ead
fdf["stressed_EL"] = fdf["stressed_prob"] * lgd * ead
fdf["stressed_bucket"] = pd.cut(fdf["stressed_prob"], bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])

# =============================================
# PAGE: OVERVIEW
# =============================================
if page == "Overview":
    st.markdown("""
    <div class="page-header">
        <span class="page-title">Portfolio Overview</span>
        <span class="page-badge"><span class="live-dot"></span>LIVE MONITORING</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner">
        This dashboard monitors <b>loan portfolio health</b> in real time. Think of it like a health checkup for a bank's loans —
        it shows how many borrowers are risky, how much money could be lost, and helps the bank prepare for tough economic times.
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    total = len(fdf)
    defaults = fdf["target"].sum()
    dr = fdf["target"].mean() * 100 if total > 0 else 0
    avg_score = fdf["prob"].mean() * 100 if total > 0 else 0
    total_el = fdf["EL"].sum()
    high_pct = (fdf["risk_bucket"] == "High").mean() * 100 if total > 0 else 0

    c1.metric("Total Loans", f"{total:,}")
    c2.metric("Defaults", f"{defaults:,.0f}")
    c3.metric("Default Rate", f"{dr:.1f}%")
    c4.metric("Avg Risk Score", f"{avg_score:.1f}%")
    c5.metric("Expected Loss", f"Rs {total_el:,.0f}")
    c6.metric("High Risk", f"{high_pct:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    r1c1, r1c2, r1c3 = st.columns([1, 1, 1])

    with r1c1:
        st.markdown('<div class="card"><div class="card-title">Risk Distribution</div><div class="card-desc">Loans split by risk category</div></div>', unsafe_allow_html=True)
        bc = fdf["risk_bucket"].value_counts().reset_index()
        bc.columns = ["Category", "Count"]
        fig = px.pie(bc, names="Category", values="Count", color="Category", color_discrete_map=RISK_COLORS, hole=0.55)
        fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=True, legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"))
        fig.update_traces(textposition="inside", textinfo="percent", textfont_size=13, marker=dict(line=dict(color="#0a0e27", width=2)))
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.markdown('<div class="card"><div class="card-title">Default Rate by Risk</div><div class="card-desc">What % of each group actually defaulted</div></div>', unsafe_allow_html=True)
        bd = fdf.groupby("risk_bucket")["target"].mean().reset_index()
        bd.columns = ["Category", "Rate"]
        bd["Rate%"] = bd["Rate"] * 100
        fig2 = px.bar(bd, x="Category", y="Rate%", color="Category", color_discrete_map=RISK_COLORS, text_auto=".1f")
        fig2.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False, yaxis_title="Default Rate %")
        fig2.update_traces(textposition="outside", textfont=dict(size=13, color="#e8edf5"), marker_line_width=0)
        st.plotly_chart(fig2, use_container_width=True)

    with r1c3:
        st.markdown('<div class="card"><div class="card-title">Expected Loss Breakdown</div><div class="card-desc">Potential monetary loss per category</div></div>', unsafe_allow_html=True)
        el_b = fdf.groupby("risk_bucket")["EL"].sum().reset_index()
        el_b.columns = ["Category", "EL"]
        fig3 = px.bar(el_b, x="Category", y="EL", color="Category", color_discrete_map=RISK_COLORS, text_auto=",.0f")
        fig3.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False, yaxis_title="Expected Loss (Rs)")
        fig3.update_traces(textposition="outside", textfont=dict(size=11, color="#e8edf5"), marker_line_width=0)
        st.plotly_chart(fig3, use_container_width=True)

    # Charts row 2
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown('<div class="card"><div class="card-title">Risk Score Distribution</div><div class="card-desc">How risk scores are spread across the portfolio</div></div>', unsafe_allow_html=True)
        fig4 = px.histogram(fdf, x="prob", nbins=60, color="risk_bucket", color_discrete_map=RISK_COLORS, barmode="stack", labels={"prob": "Risk Score"})
        fig4.update_layout(**PLOTLY_LAYOUT, height=340, legend_title_text="")
        st.plotly_chart(fig4, use_container_width=True)

    with r2c2:
        st.markdown('<div class="card"><div class="card-title">Risk Score vs Interest Rate</div><div class="card-desc">Higher interest usually means higher risk</div></div>', unsafe_allow_html=True)
        sample = fdf.sample(min(2000, len(fdf)), random_state=42)
        fig5 = px.scatter(sample, x="int_rate", y="prob", color="risk_bucket", color_discrete_map=RISK_COLORS, opacity=0.5, labels={"int_rate": "Interest Rate (WOE)", "prob": "Risk Score"})
        fig5.update_layout(**PLOTLY_LAYOUT, height=340, legend_title_text="")
        st.plotly_chart(fig5, use_container_width=True)


# =============================================
# PAGE: RISK ANALYSIS
# =============================================
elif page == "Risk Analysis":
    st.markdown("""
    <div class="page-header">
        <span class="page-title">Risk Analysis</span>
        <span class="page-badge">DEEP DIVE</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner">
        Explore individual features to understand what makes a loan risky. Pick a feature below and see how it relates to default risk.
    </div>
    """, unsafe_allow_html=True)

    feat_cols = [c for c in fdf.columns if c not in ["target", "prob", "risk_bucket", "EL", "stressed_prob", "stressed_EL", "stressed_bucket"]]
    sel_feat = st.selectbox("Select Feature", feat_cols, format_func=lambda x: FRIENDLY.get(x, x))

    explanations = {
        "grade": "A rating given to each loan — higher grade = safer, lower grade = riskier.",
        "int_rate": "How much extra the borrower pays. Higher interest rate = bank thinks the borrower is riskier.",
        "all_util": "What % of total available credit is being used. Using too much credit = risky behavior.",
        "max_bal_bc": "The biggest balance on any credit card. Bigger balances can mean more financial strain.",
        "mths_since_rcnt_il": "How recently they took another loan. Very recent loans could signal financial trouble.",
        "total_bal_il": "Total amount owed on all installment loans. More debt = higher risk.",
        "il_util": "What % of installment credit is being used up. Higher utilization = potential trouble.",
    }
    if sel_feat in explanations:
        st.markdown(f'<div class="info-banner"><b>{FRIENDLY.get(sel_feat, sel_feat)}:</b> {explanations[sel_feat]}</div>', unsafe_allow_html=True)

    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown(f'<div class="card"><div class="card-title">Distribution by Risk</div><div class="card-desc">How {FRIENDLY.get(sel_feat, sel_feat)} is spread across risk categories</div></div>', unsafe_allow_html=True)
        fig = px.histogram(fdf, x=sel_feat, nbins=40, color="risk_bucket", color_discrete_map=RISK_COLORS, barmode="overlay", opacity=0.65, labels={sel_feat: FRIENDLY.get(sel_feat, sel_feat)})
        fig.update_layout(**PLOTLY_LAYOUT, height=380, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with ac2:
        st.markdown(f'<div class="card"><div class="card-title">{FRIENDLY.get(sel_feat, sel_feat)} vs Risk Score</div><div class="card-desc">Scatter plot showing relationship</div></div>', unsafe_allow_html=True)
        s = fdf.sample(min(2500, len(fdf)), random_state=42)
        fig2 = px.scatter(s, x=sel_feat, y="prob", color="risk_bucket", color_discrete_map=RISK_COLORS, opacity=0.45, labels={sel_feat: FRIENDLY.get(sel_feat, sel_feat), "prob": "Risk Score"})
        fig2.update_layout(**PLOTLY_LAYOUT, height=380, legend_title_text="")
        st.plotly_chart(fig2, use_container_width=True)

    # Box plot
    st.markdown(f'<div class="card"><div class="card-title">Comparison Across Risk Buckets</div><div class="card-desc">Box plot of {FRIENDLY.get(sel_feat, sel_feat)} segmented by risk</div></div>', unsafe_allow_html=True)
    fig3 = px.box(fdf, x="risk_bucket", y=sel_feat, color="risk_bucket", color_discrete_map=RISK_COLORS, labels={"risk_bucket": "Risk Category", sel_feat: FRIENDLY.get(sel_feat, sel_feat)})
    fig3.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    # Stats table
    st.markdown('<div class="card"><div class="card-title">Summary Statistics</div></div>', unsafe_allow_html=True)
    stats = fdf.groupby("risk_bucket")[sel_feat].describe().round(3)
    st.dataframe(stats, use_container_width=True)


# =============================================
# PAGE: STRESS TESTING
# =============================================
elif page == "Stress Testing":
    st.markdown("""
    <div class="page-header">
        <span class="page-title">Stress Testing</span>
        <span class="page-badge">SCENARIO ANALYSIS</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner">
        <b>What is Stress Testing?</b> Imagine the economy crashes (recession, pandemic, etc). People lose jobs and can't repay loans.
        Stress testing simulates this by <b>increasing everyone's risk score</b> to see how much extra money the bank could lose.
        Use the <b>Stress Multiplier</b> in the sidebar (set it above 1.0x).
    </div>
    """, unsafe_allow_html=True)

    if stress > 1.0:
        normal_el = fdf["EL"].sum()
        stressed_el = fdf["stressed_EL"].sum()
        el_change = ((stressed_el - normal_el) / normal_el * 100) if normal_el > 0 else 0
        normal_dr = fdf["target"].mean() * 100
        stressed_dr = fdf["stressed_prob"].mean() * 100

        st.markdown(f"### Scenario: {stress}x Stress Applied")

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Normal Exp. Loss", f"Rs {normal_el:,.0f}")
        sc2.metric("Stressed Exp. Loss", f"Rs {stressed_el:,.0f}")
        sc3.metric("Loss Increase", f"+{el_change:.1f}%")
        sc4.metric("Extra Capital Needed", f"Rs {stressed_el - normal_el:,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Migration
        st.markdown('<div class="card"><div class="card-title">Risk Migration Matrix</div><div class="card-desc">How loans move between risk categories under stress</div></div>', unsafe_allow_html=True)
        migration = pd.crosstab(fdf["risk_bucket"], fdf["stressed_bucket"], margins=True, margins_name="Total")
        st.dataframe(migration, use_container_width=True)

        # Before vs After
        st.markdown("<br>", unsafe_allow_html=True)
        bc1, bc2 = st.columns(2)

        with bc1:
            st.markdown('<div class="card"><div class="card-title">Before Stress</div></div>', unsafe_allow_html=True)
            nc = fdf["risk_bucket"].value_counts().reset_index()
            nc.columns = ["Category", "Count"]
            fig = px.pie(nc, names="Category", values="Count", color="Category", color_discrete_map=RISK_COLORS, hole=0.5)
            fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=True)
            fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=12, marker=dict(line=dict(color="#0a0e27", width=2)))
            st.plotly_chart(fig, use_container_width=True)

        with bc2:
            st.markdown('<div class="card"><div class="card-title">After Stress</div></div>', unsafe_allow_html=True)
            sc = fdf["stressed_bucket"].value_counts().reset_index()
            sc.columns = ["Category", "Count"]
            fig2 = px.pie(sc, names="Category", values="Count", color="Category", color_discrete_map=RISK_COLORS, hole=0.5)
            fig2.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=True)
            fig2.update_traces(textposition="inside", textinfo="percent+label", textfont_size=12, marker=dict(line=dict(color="#0a0e27", width=2)))
            st.plotly_chart(fig2, use_container_width=True)

        # EL comparison bar
        st.markdown('<div class="card"><div class="card-title">Expected Loss Comparison</div><div class="card-desc">Side-by-side: Normal vs Stressed expected loss per risk bucket</div></div>', unsafe_allow_html=True)
        el_norm = fdf.groupby("risk_bucket")["EL"].sum().reset_index()
        el_norm.columns = ["Category", "Loss"]
        el_norm["Scenario"] = "Normal"
        el_str = fdf.groupby("risk_bucket")["stressed_EL"].sum().reset_index()
        el_str.columns = ["Category", "Loss"]
        el_str["Scenario"] = "Stressed"
        el_comp = pd.concat([el_norm, el_str])
        fig3 = px.bar(el_comp, x="Category", y="Loss", color="Scenario", barmode="group",
                      color_discrete_map={"Normal": ACCENT, "Stressed": "#ef4444"}, text_auto=",.0f")
        fig3.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_title="Expected Loss (Rs)")
        fig3.update_traces(textposition="outside", textfont=dict(size=11, color="#e8edf5"), marker_line_width=0)
        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 80px 20px;">
            <div style="font-size: 3.5rem; margin-bottom: 16px;">🌤</div>
            <div style="color: #e8edf5; font-size: 1.2rem; font-weight: 700;">No Stress Applied</div>
            <div style="color: #5a6a8a; font-size: 0.85rem; margin-top: 8px;">
                Move the <b>Stress Multiplier</b> in the sidebar above 1.0x to simulate a recession.
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================
# PAGE: FEATURE INTELLIGENCE
# =============================================
elif page == "Feature Intelligence":
    st.markdown("""
    <div class="page-header">
        <span class="page-title">Feature Intelligence</span>
        <span class="page-badge">AI INSIGHTS</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner">
        <b>Features</b> are the data points the model uses to make predictions — like clues that help solve a puzzle.
        This page shows which clues matter most and how they relate to each other.
    </div>
    """, unsafe_allow_html=True)

    num_feats = [c for c in ["grade", "int_rate", "all_util", "max_bal_bc", "mths_since_rcnt_il", "total_bal_il", "il_util", "prob"] if c in fdf.columns]

    # Correlation heatmap
    st.markdown('<div class="card"><div class="card-title">Correlation Matrix</div><div class="card-desc">Shows how strongly features are connected — darker red = stronger positive link, darker blue = stronger negative link</div></div>', unsafe_allow_html=True)

    corr = fdf[num_feats].corr()
    labels = [FRIENDLY.get(c, c) for c in corr.columns]
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale=[[0, "#2563eb"], [0.5, "#0a0e27"], [1, "#ef4444"]],
        text=np.round(corr.values, 2), texttemplate="%{text}", textfont={"size": 11, "color": "#8fa3c4"},
        hoverongaps=False,
        colorbar=dict(title="r", titlefont=dict(color="#8fa3c4"), tickfont=dict(color="#8fa3c4"))
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=480, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.markdown('<div class="card"><div class="card-title">Feature Importance (Correlation with Default)</div><div class="card-desc">Which features have the strongest link to loan default?</div></div>', unsafe_allow_html=True)

    if "target" in fdf.columns:
        tc = fdf[num_feats].corrwith(fdf["target"]).abs().sort_values(ascending=True).reset_index()
        tc.columns = ["Feature", "Importance"]
        tc["Name"] = tc["Feature"].map(FRIENDLY)
        fig2 = px.bar(tc, x="Importance", y="Name", orientation="h", color="Importance",
                      color_continuous_scale=[[0, "#1e3a5f"], [0.5, ACCENT], [1, "#22c55e"]], text_auto=".3f")
        fig2.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False, yaxis_title="", coloraxis_showscale=False)
        fig2.update_traces(textposition="outside", textfont=dict(size=12, color="#e8edf5"), marker_line_width=0)
        st.plotly_chart(fig2, use_container_width=True)

    # Glossary
    st.markdown('<div class="card"><div class="card-title">Feature Glossary</div><div class="card-desc">Plain English explanations of each feature</div></div>', unsafe_allow_html=True)
    glossary = {
        "Loan Grade": "A safety rating for the loan. Better grades = safer loans.",
        "Interest Rate": "The extra cost of borrowing. Higher rates mean the bank thinks you're riskier.",
        "Credit Utilization": "How much of your credit limit you're using. Using 90% of your limit? Risky!",
        "Max Card Balance": "The biggest amount owed on any single credit card.",
        "Months Since Last Installment": "How recently you took another installment loan.",
        "Total Installment Balance": "Total of all installment debts — car loans, personal loans, etc.",
        "Installment Utilization": "How much of your installment credit capacity is being used.",
        "Risk Score": "The model's final prediction — probability that this person will default."
    }
    for name, desc in glossary.items():
        st.markdown(f"""<div class="stat-row" style="margin-bottom:8px;">
            <span class="stat-label" style="font-weight:700; color:#e8edf5;">{name}</span>
            <span class="stat-value" style="font-weight:400; color:#7c8db5; font-size:0.75rem; text-align:right;">{desc}</span>
        </div>""", unsafe_allow_html=True)


# =============================================
# PAGE: DATA EXPLORER
# =============================================
elif page == "Data Explorer":
    st.markdown("""
    <div class="page-header">
        <span class="page-title">Data Explorer</span>
        <span class="page-badge">RAW DATA</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-banner">
        Browse, search, and download the dataset. Currently showing <b>{len(fdf):,}</b> loans after all filters applied.
    </div>
    """, unsafe_allow_html=True)

    # Summary stats
    st.markdown('<div class="card"><div class="card-title">Summary Statistics</div></div>', unsafe_allow_html=True)
    desc = fdf[["grade", "int_rate", "all_util", "max_bal_bc", "total_bal_il", "prob"]].describe().round(3)
    desc.columns = [FRIENDLY.get(c, c) for c in desc.columns]
    st.dataframe(desc, use_container_width=True)

    # Full data
    st.markdown('<div class="card"><div class="card-title">Full Dataset</div></div>', unsafe_allow_html=True)
    display = fdf.rename(columns=FRIENDLY)
    st.dataframe(display, use_container_width=True, height=500)

    # Download
    csv = fdf.to_csv(index=False).encode("utf-8")
    st.download_button("Download Filtered Data (CSV)", csv, "lenders_club_filtered.csv", "text/csv")


# =============================================
# FOOTER
# =============================================
st.markdown("""
<div class="footer-bar">
    <div class="footer-text">
        LENDER'S CLUB RISK MONITOR &nbsp;&bull;&nbsp; BUILT WITH STREAMLIT + PLOTLY &nbsp;&bull;&nbsp; DATA-DRIVEN DECISION MAKING
    </div>
</div>
""", unsafe_allow_html=True)
