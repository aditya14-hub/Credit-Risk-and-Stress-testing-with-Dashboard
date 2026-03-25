import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Institutional Risk Monitor",
    layout="wide",
    page_icon="LC",
    initial_sidebar_state="expanded"
)

# =============================================
# INSTITUTIONAL BANKING CSS
# =============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ===== GLOBAL ===== */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #050505 !important; } /* Pure Jet Black */

/* Fix: Keep header active so sidebar toggle works, but hide ALL top-right icons (deploy, share, github) */
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
header { background: transparent !important; }

/* Hide exactly the right-side header icons without breaking the left-side sidebar toggle */
[data-testid="stHeaderActionElements"] { display: none !important; }
.stAppDeployButton { display: none !important; }
[class^="viewerBadge"] { display: none !important; }

/* Hide the "Manage App" watermark at the bottom right */
[data-testid="manage-app-button"] { display: none !important; }

/* Make sure the sidebar toggle icon is strictly visible */
[data-testid="collapsedControl"] {
    color: #FFFFFF !important;
    background: #0F0F0F !important;
    border-radius: 4px;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #050505; }
::-webkit-scrollbar-thumb { background: #4361EE; border-radius: 0px; } /* Enterprise Blue */

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: #0A0A0A !important; /* Extremely dark grey */
    border-right: 1px solid #1A1A1A;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* ===== KPI METRIC CARDS ===== */
div[data-testid="stMetric"] {
    background: #0F0F0F;
    border: 1px solid #222222;
    border-radius: 2px; /* Ultra sharp corners */
    padding: 18px 14px;
    box-shadow: none; /* Removed shadow for flat, sharp look */
    transition: all 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    border-color: #4361EE; /* Sharp blue border on hover, NO bouncing */
}
div[data-testid="stMetric"] label {
    color: #888888 !important; /* Muted Flat Gray */
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}
div[data-testid="stMetric"] label > div { white-space: normal !important; overflow: visible !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #FFFFFF !important; /* Pure White */
    font-weight: 700 !important;
    font-size: 1.4rem !important;
    white-space: normal !important;
    overflow: visible !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] > div { white-space: normal !important; overflow: visible !important; }
div[data-testid="stMetric"] [data-testid="stMetricDelta"] svg { display: none; }

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    background: #0F0F0F;
    border-radius: 2px;
    padding: 2px;
    gap: 2px;
    border: 1px solid #222222;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 2px;
    color: #888888 !important;
    font-weight: 600;
    font-size: 0.8rem;
    padding: 6px 14px;
    letter-spacing: 0.2px;
}
.stTabs [aria-selected="true"] {
    background: #4361EE !important; /* Sharp Enterprise Blue */
    color: #ffffff !important;
}

/* ===== CUSTOM COMPONENTS ===== */
.brand-bar {
    background: #050505;
    border-bottom: 1px solid #1A1A1A;
    padding: 24px 20px;
    margin: -1rem -1rem 20px -1rem;
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-logo {
    width: 34px; height: 34px;
    background: #4361EE;
    border-radius: 2px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; color: white; font-weight: 700;
}
.brand-text { color: #FFFFFF; font-size: 0.95rem; font-weight: 600; letter-spacing: 0.5px; }
.brand-sub { color: #888888; font-size: 0.65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }

.nav-section { color: #555555; font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin: 24px 0 8px 0; padding: 0 4px; }

.stat-row {
    display: flex; gap: 8px; padding: 10px 12px;
    border-radius: 2px; margin: 4px 0;
    background: #0F0F0F;
    border: 1px solid #222222;
}
.stat-label { color: #888888; font-size: 0.7rem; font-weight: 500; flex: 1; }
.stat-value { color: #FFFFFF; font-size: 0.75rem; font-weight: 600; }

.page-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 24px; padding-bottom: 16px;
    border-bottom: 1px solid #222222;
}
.page-title { color: #FFFFFF; font-size: 1.4rem; font-weight: 600; letter-spacing: 0.5px; }
.page-badge {
    background: rgba(67, 97, 238, 0.1);
    border: 1px solid #4361EE;
    color: #4361EE; font-size: 0.65rem; font-weight: 600;
    padding: 4px 12px; border-radius: 2px;
    letter-spacing: 0.8px;
    display: flex; align-items: center; gap: 6px;
}
.live-dot {
    width: 6px; height: 6px; background: #00E676; border-radius: 50%;
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0, 230, 118, 0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 4px rgba(0, 230, 118, 0); }
}

.card {
    background: #0F0F0F;
    border: 1px solid #222222;
    border-radius: 2px;
    padding: 24px;
    margin-bottom: 20px;
}
.card-title {
    color: #FFFFFF; font-size: 0.9rem; font-weight: 600; letter-spacing: 0.5px;
    margin-bottom: 4px; display: flex; align-items: center; gap: 8px;
}
.card-desc { color: #888888; font-size: 0.75rem; font-weight: 400; margin-bottom: 16px; }

.info-banner {
    background: rgba(67, 97, 238, 0.05);
    border-left: 3px solid #4361EE;
    padding: 14px 18px;
    color: #CCCCCC; font-size: 0.8rem; line-height: 1.6;
    margin-bottom: 20px;
}

.footer-bar {
    text-align: center; padding: 30px 0 10px;
    border-top: 1px solid #222222;
    margin-top: 40px;
}
.footer-text { color: #555555; font-size: 0.65rem; font-weight: 500; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

# =============================================
# PLOTLY THEME
# =============================================
PLOTLY_BG = "rgba(0,0,0,0)"
PLOTLY_LAYOUT = dict(
    paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
    font=dict(family="Inter", color="#888888", size=11),
    margin=dict(l=30, r=20, t=40, b=30),
    xaxis=dict(gridcolor="#1A1A1A", zerolinecolor="#222222"),
    yaxis=dict(gridcolor="#1A1A1A", zerolinecolor="#222222"),
    hoverlabel=dict(bgcolor="#0F0F0F", font_size=12, font_family="Inter", bordercolor="#4361EE"),
)
# Finance Color Palette (Sharp Green, Amber, Red)
RISK_COLORS = {"Low": "#00E676", "Medium": "#FFAB00", "High": "#FF1744"}
ACCENT = "#4361EE" # Enterprise Blue

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
# SIDEBAR — Institutional Navigation
# =============================================
with st.sidebar:
    st.markdown("""
    <div class="brand-bar">
        <div class="brand-logo">LC</div>
        <div>
            <div class="brand-text">Lender's Club</div>
            <div class="brand-sub">Institutional Risk</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "Go to",
        ["Overview", "Risk Segmentation", "Stress Testing", "Model Insight"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="nav-section">Portfolio Filters</div>', unsafe_allow_html=True)
    risk_opts = sorted(df["risk_bucket"].dropna().unique().tolist())
    sel_risk = st.multiselect("Risk Category", risk_opts, default=risk_opts)

    prob_min, prob_max = float(df["prob"].min()), float(df["prob"].max())
    prob_range = st.slider("Risk Score Tolerance", prob_min, prob_max, (prob_min, prob_max), format="%.2f")

    st.markdown('<div class="nav-section">Macroeconomic Stress</div>', unsafe_allow_html=True)
    stress = st.slider("PD Stress Multiplier", 1.0, 2.5, 1.0, 0.1)
    lgd = st.slider("Loss Given Default (LGD %)", 10, 100, 60, 5) / 100
    ead = st.number_input("Average EAD (Rs)", value=100000, step=10000)

    st.markdown('<div class="nav-section">Portfolio Status</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-row"><span class="stat-label">Total Facilities</span><span class="stat-value">{len(df):,}</span></div>
    <div class="stat-row"><span class="stat-label">Model Features</span><span class="stat-value">{len(df.columns)}</span></div>
    <div class="stat-row"><span class="stat-label">Default Rate</span><span class="stat-value">{df['target'].mean()*100:.1f}%</span></div>
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
# FORMATTER HELPER
# =============================================
def fmt_val(val, is_currency=False):
    if val >= 1_000_000_000: return f"Rs {val/1_000_000_000:.1f}B" if is_currency else f"{val/1_000_000_000:.1f}B"
    if val >= 1_000_000: return f"Rs {val/1_000_000:.1f}M" if is_currency else f"{val/1_000_000:.1f}M"
    if val >= 1_000: return f"Rs {val/1_000:.1f}K" if is_currency else f"{val/1_000:.1f}K"
    return f"Rs {val:,.0f}" if is_currency else f"{val:,.0f}"

# =============================================
# PAGE: OVERVIEW
# =============================================
if page == "Overview":
    st.markdown("""
    <div class="page-header">
        <span class="page-title">Portfolio Overview</span>
        <span class="page-badge"><span class="live-dot"></span>ACTIVE</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner">
        High-level monitoring of retail credit risk exposures. Metrics track predictive risk factors and expected capital loss.
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

    c1.metric("Total Facilities", fmt_val(total))
    c2.metric("Gross Defaults", fmt_val(defaults))
    c3.metric("Observed PD", f"{dr:.1f}%")
    c4.metric("Avg Risk Score", f"{avg_score:.1f}%")
    c5.metric("Expected Loss", fmt_val(total_el, True))
    c6.metric("High Risk Tier", f"{high_pct:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    r1c1, r1c2, r1c3 = st.columns([1, 1, 1])

    with r1c1:
        st.markdown('<div class="card"><div class="card-title">Exposure by Risk Tier</div><div class="card-desc">Volume of loans categorized by predictive PD bucket</div></div>', unsafe_allow_html=True)
        bc = fdf["risk_bucket"].value_counts().reset_index()
        bc.columns = ["Category", "Count"]
        fig = px.pie(bc, names="Category", values="Count", color="Category", color_discrete_map=RISK_COLORS, hole=0.6)
        fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=True, legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"))
        fig.update_traces(textposition="inside", textinfo="percent", textfont_size=11, marker=dict(line=dict(color="#141B2D", width=2)))
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.markdown('<div class="card"><div class="card-title">Default Prevalence</div><div class="card-desc">Actual default occurrence % by tier</div></div>', unsafe_allow_html=True)
        bd = fdf.groupby("risk_bucket")["target"].mean().reset_index()
        bd.columns = ["Category", "Rate"]
        bd["Rate%"] = bd["Rate"] * 100
        fig2 = px.bar(bd, x="Category", y="Rate%", color="Category", color_discrete_map=RISK_COLORS, text_auto=".1f")
        fig2.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False, yaxis_title="Default Rate %")
        fig2.update_traces(textposition="outside", textfont=dict(size=11, color="#F3F4F6"), marker_line_width=0)
        st.plotly_chart(fig2, use_container_width=True)

    with r1c3:
        st.markdown('<div class="card"><div class="card-title">Expected Credit Loss</div><div class="card-desc">Simulated fiscal impact (PD x LGD x EAD)</div></div>', unsafe_allow_html=True)
        el_b = fdf.groupby("risk_bucket")["EL"].sum().reset_index()
        el_b.columns = ["Category", "EL"]
        fig3 = px.bar(el_b, x="Category", y="EL", color="Category", color_discrete_map=RISK_COLORS, text_auto=",.0f")
        fig3.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False, yaxis_title="EL (Rs)")
        fig3.update_traces(textposition="outside", textfont=dict(size=11, color="#F3F4F6"), marker_line_width=0)
        st.plotly_chart(fig3, use_container_width=True)

    # Charts row 2
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown('<div class="card"><div class="card-title">Risk Score Density</div><div class="card-desc">Granular distribution of predictive probabilities</div></div>', unsafe_allow_html=True)
        fig4 = px.histogram(fdf, x="prob", nbins=60, color="risk_bucket", color_discrete_map=RISK_COLORS, barmode="stack", labels={"prob": "Probability of Default (PD)"})
        fig4.update_layout(**PLOTLY_LAYOUT, height=340, legend_title_text="")
        st.plotly_chart(fig4, use_container_width=True)

    with r2c2:
        st.markdown('<div class="card"><div class="card-title">Yield vs. Risk Alignment</div><div class="card-desc">Assessing pricing efficiency (Interest Rate vs PD)</div></div>', unsafe_allow_html=True)
        sample = fdf.sample(min(2000, len(fdf)), random_state=42)
        fig5 = px.scatter(sample, x="int_rate", y="prob", color="risk_bucket", color_discrete_map=RISK_COLORS, opacity=0.5, labels={"int_rate": "Interest Rate (Factor)", "prob": "PD Score"})
        fig5.update_layout(**PLOTLY_LAYOUT, height=340, legend_title_text="")
        st.plotly_chart(fig5, use_container_width=True)

# =============================================
# PAGE: RISK SEGMENTATION
# =============================================
elif page == "Risk Segmentation":
    st.markdown("""
    <div class="page-header">
        <span class="page-title">Risk Segmentation Analysis</span>
        <span class="page-badge">DRILL-DOWN</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner">
        Evaluate how specific borrower characteristics map to capital risk. Select an encoded feature to analyze its underlying distributional impact.
    </div>
    """, unsafe_allow_html=True)

    feat_cols = [c for c in fdf.columns if c not in ["target", "prob", "risk_bucket", "EL", "stressed_prob", "stressed_EL", "stressed_bucket"]]
    sel_feat = st.selectbox("Select Characteristic", feat_cols, format_func=lambda x: FRIENDLY.get(x, x))

    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown(f'<div class="card"><div class="card-title">Distribution across Tiers</div><div class="card-desc">Density overlay by {FRIENDLY.get(sel_feat, sel_feat)}</div></div>', unsafe_allow_html=True)
        fig = px.histogram(fdf, x=sel_feat, nbins=40, color="risk_bucket", color_discrete_map=RISK_COLORS, barmode="overlay", opacity=0.6, labels={sel_feat: FRIENDLY.get(sel_feat, sel_feat)})
        fig.update_layout(**PLOTLY_LAYOUT, height=380, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with ac2:
        st.markdown(f'<div class="card"><div class="card-title">Score Mapping</div><div class="card-desc">Scatter mapping against final PD Score</div></div>', unsafe_allow_html=True)
        s = fdf.sample(min(2500, len(fdf)), random_state=42)
        fig2 = px.scatter(s, x=sel_feat, y="prob", color="risk_bucket", color_discrete_map=RISK_COLORS, opacity=0.4, labels={sel_feat: FRIENDLY.get(sel_feat, sel_feat), "prob": "PD Score"})
        fig2.update_layout(**PLOTLY_LAYOUT, height=380, legend_title_text="")
        st.plotly_chart(fig2, use_container_width=True)

    # Box plot
    st.markdown(f'<div class="card"><div class="card-title">Volatility Analysis</div><div class="card-desc">Box plot variance of {FRIENDLY.get(sel_feat, sel_feat)} by tier</div></div>', unsafe_allow_html=True)
    fig3 = px.box(fdf, x="risk_bucket", y=sel_feat, color="risk_bucket", color_discrete_map=RISK_COLORS, labels={"risk_bucket": "Tier", sel_feat: FRIENDLY.get(sel_feat, sel_feat)})
    fig3.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

# =============================================
# PAGE: STRESS TESTING
# =============================================
elif page == "Stress Testing":
    st.markdown("""
    <div class="page-header">
        <span class="page-title">Macroeconomic Stress Simulation</span>
        <span class="page-badge">BASEL III SIMULATOR</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner">
        <b>Adverse Scenario Modeling:</b> Simulates portfolio shocks by applying a uniform multiplier to baseline PDs. 
        Calculates resulting Expected Credit Loss (ECL) variance and tier migration matrices. (Configure via sidebar).
    </div>
    """, unsafe_allow_html=True)

    if stress > 1.0:
        normal_el = fdf["EL"].sum()
        stressed_el = fdf["stressed_EL"].sum()
        el_change = ((stressed_el - normal_el) / normal_el * 100) if normal_el > 0 else 0

        st.markdown(f"### Economic Shock Scenario: {stress}x Baseline Risk")

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Baseline EL", fmt_val(normal_el, True))
        sc2.metric("Stressed EL", fmt_val(stressed_el, True))
        sc3.metric("Capital Impact", f"+{el_change:.1f}%")
        sc4.metric("Capital Shortfall", fmt_val(stressed_el - normal_el, True))

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">Tier Migration Matrix</div><div class="card-desc">Facility transition volumes under stressed scenario</div></div>', unsafe_allow_html=True)
        migration = pd.crosstab(fdf["risk_bucket"], fdf["stressed_bucket"], margins=True, margins_name="Total")
        st.dataframe(migration, use_container_width=True)

        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown('<div class="card"><div class="card-title">Baseline Portfolio</div></div>', unsafe_allow_html=True)
            nc = fdf["risk_bucket"].value_counts().reset_index()
            nc.columns = ["Tier", "Count"]
            fig = px.pie(nc, names="Tier", values="Count", color="Tier", color_discrete_map=RISK_COLORS, hole=0.55)
            fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=True)
            fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=11, marker=dict(line=dict(color="#141B2D", width=2)))
            st.plotly_chart(fig, use_container_width=True)

        with bc2:
            st.markdown('<div class="card"><div class="card-title">Stressed Portfolio</div></div>', unsafe_allow_html=True)
            sc = fdf["stressed_bucket"].value_counts().reset_index()
            sc.columns = ["Tier", "Count"]
            fig2 = px.pie(sc, names="Tier", values="Count", color="Tier", color_discrete_map=RISK_COLORS, hole=0.55)
            fig2.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=True)
            fig2.update_traces(textposition="inside", textinfo="percent+label", textfont_size=11, marker=dict(line=dict(color="#141B2D", width=2)))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="card"><div class="card-title">Expected Loss Variance</div><div class="card-desc">Comparative analysis of credit provisions</div></div>', unsafe_allow_html=True)
        el_norm = fdf.groupby("risk_bucket")["EL"].sum().reset_index()
        el_norm.columns = ["Tier", "Loss"]
        el_norm["Condition"] = "Baseline"
        el_str = fdf.groupby("risk_bucket")["stressed_EL"].sum().reset_index()
        el_str.columns = ["Tier", "Loss"]
        el_str["Condition"] = "Stressed"
        fig3 = px.bar(pd.concat([el_norm, el_str]), x="Tier", y="Loss", color="Condition", barmode="group",
                      color_discrete_map={"Baseline": ACCENT, "Stressed": "#EF4444"}, text_auto=",.0f")
        fig3.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_title="EL (Rs)")
        fig3.update_traces(textposition="outside", textfont=dict(size=10, color="#F3F4F6"), marker_line_width=0)
        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 80px 20px;">
            <div style="color: #F3F4F6; font-size: 1.2rem; font-weight: 600;">Standard Economic Conditions</div>
            <div style="color: #9CA3AF; font-size: 0.85rem; margin-top: 8px;">
                Increase the <b>Stress Multiplier</b> constraint in the sidebar to run simulations.
            </div>
        </div>
        """, unsafe_allow_html=True)

# =============================================
# PAGE: MODEL INSIGHT
# =============================================
elif page == "Model Insight":
    st.markdown("""
    <div class="page-header">
        <span class="page-title">Predictive Model Insights</span>
        <span class="page-badge">XGBoost Diagnostics</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner">
        Mathematical interpretability of the underlying classification model. Analyzing multicollinearity and individual feature significance in predicting default outcomes.
    </div>
    """, unsafe_allow_html=True)

    num_feats = [c for c in ["grade", "int_rate", "all_util", "max_bal_bc", "mths_since_rcnt_il", "total_bal_il", "il_util", "prob"] if c in fdf.columns]

    st.markdown('<div class="card"><div class="card-title">Feature Collinearity Matrix</div><div class="card-desc">Pearson correlation coefficients (r) between encoded inputs</div></div>', unsafe_allow_html=True)
    corr = fdf[num_feats].corr()
    labels = [FRIENDLY.get(c, c) for c in corr.columns]
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale=[[0, "#0F766E"], [0.5, "#0A0F1C"], [1, "#D97706"]],
        text=np.round(corr.values, 2), texttemplate="%{text}", textfont={"size": 10, "color": "#9CA3AF"},
        colorbar=dict(title=dict(text="r", font=dict(color="#9CA3AF")), tickfont=dict(color="#9CA3AF"))
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=480, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="card"><div class="card-title">Absolute Feature Significance</div><div class="card-desc">Ranked proxy importance based on target correlation magnitude</div></div>', unsafe_allow_html=True)
    if "target" in fdf.columns:
        tc = fdf[num_feats].corrwith(fdf["target"]).abs().sort_values(ascending=True).reset_index()
        tc.columns = ["Feature", "Importance"]
        tc["Name"] = tc["Feature"].map(FRIENDLY)
        fig2 = px.bar(tc, x="Importance", y="Name", orientation="h", color="Importance",
                      color_continuous_scale=[[0, "#141B2D"], [0.5, "#0F766E"], [1, "#10B981"]], text_auto=".3f")
        fig2.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False, yaxis_title="", coloraxis_showscale=False)
        fig2.update_traces(textposition="outside", textfont=dict(size=11, color="#F3F4F6"), marker_line_width=0)
        st.plotly_chart(fig2, use_container_width=True)

# =============================================
# FOOTER
# =============================================
st.markdown("""
<div class="footer-bar">
    <div class="footer-text">
        LENDER'S CLUB INSTITUTIONAL RISK &nbsp;&bull;&nbsp; PROPRIETARY ANALYTICS CAPABILITY
    </div>
</div>
""", unsafe_allow_html=True)
