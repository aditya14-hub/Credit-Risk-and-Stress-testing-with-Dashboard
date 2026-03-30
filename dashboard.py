import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import logging
from typing import Dict, Any, List

# ==============================================================================
# CONFIGURATION & INITIALIZATION
# ==============================================================================
st.set_page_config(
    page_title="Stratos Analytics — Institutional Risk Core",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# THEME DEFINITIONS (Lead UI/UX & Data Vis Grade)
# ==============================================================================
class Theme:
    """Centralized theme configuration for consistent CSS and Plotly styling."""
    
    # Core Palette (Deep Matrix Slate - Lowest eye strain, ultra-premium contrast)
    BG_BASE = "#050608"          # True dark (near black)
    BG_CARD = "#0C0E14"          # Deep charcoal with a touch of blue
    BG_HOVER = "#121620"         # Slight elevation
    
    TEXT_PRIMARY = "#F1F5F9"     # Bright off-white (max legibility)
    TEXT_SECONDARY = "#94A3B8"   # Slate 400 — clearly visible
    TEXT_MUTED = "#64748B"       # Slate 500 — for truly secondary info only
    
    BORDER = "#1E2A3A"           # Slightly brighter separator
    BORDER_HOVER = "#334155"     # Interaction border
    
    # Financial Severity Indicators (Desaturated, institutional tones)
    RISK_LOW = "#226B4E"         # Muted Pine Green
    RISK_MED = "#9A7332"         # Muted Bronze/Gold
    RISK_HIGH = "#8C3131"        # Deep Brick Red
    
    # Corporate Accents
    ACCENT_MAIN = "#415C8C"      # Steel Blue
    ACCENT_ALT = "#324976"       # Darker Steel
    ACCENT_GLOW = "rgba(65, 92, 140, 0.15)"
    
    @classmethod
    def get_plotly_layout(cls) -> Dict[str, Any]:
        """Returns standard Plotly layout injection for all charts."""
        return dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, -apple-system, sans-serif", color=cls.TEXT_SECONDARY, size=11),
            margin=dict(l=24, r=24, t=44, b=24),
            xaxis=dict(
                gridcolor="rgba(255,255,255,0.04)",
                zerolinecolor="rgba(255,255,255,0.06)",
                tickfont=dict(color=cls.TEXT_SECONDARY, size=11),
                title_font=dict(color=cls.TEXT_PRIMARY, size=12),
                showgrid=True,
                zeroline=False,
            ),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.04)",
                zerolinecolor="rgba(255,255,255,0.06)",
                tickfont=dict(color=cls.TEXT_SECONDARY, size=11),
                title_font=dict(color=cls.TEXT_PRIMARY, size=12),
                showgrid=True,
                zeroline=False,
            ),
            hoverlabel=dict(
                bgcolor="#1E2A3A",
                font_size=13,
                font_family="Inter",
                font_color=cls.TEXT_PRIMARY,
                bordercolor=cls.ACCENT_MAIN
            ),
            legend=dict(
                font=dict(color=cls.TEXT_PRIMARY, size=12),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            coloraxis_colorbar=dict(
                tickfont=dict(color=cls.TEXT_MUTED),
                title=dict(font=dict(color=cls.TEXT_SECONDARY)),
                bordercolor="rgba(0,0,0,0)"
            )
        )

# Inject Global CSS overriding Streamlit defaults
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif !important;
}}

/* Base Theming */
.stApp {{ background: {Theme.BG_BASE} !important; }}

/* Hide only specific Streamlit chrome — leave sidebar toggle INTACT */
#MainMenu {{ visibility: hidden !important; }}
footer {{ visibility: hidden !important; }}
[data-testid="stHeaderActionElements"] {{ visibility: hidden !important; }}
.stAppDeployButton {{ display: none !important; }}
[data-testid="manage-app-button"] {{ display: none !important; }}

/* Sidebar — style without touching toggle mechanism */
section[data-testid="stSidebar"] {{
    background: {Theme.BG_CARD} !important;
    border-right: 1px solid {Theme.BORDER};
}}

/* Sidebar toggle button — preserve Streamlit's native behavior but re-skin it */
[data-testid="collapsedControl"] {{
    background: {Theme.BG_CARD} !important;
    border: 1px solid {Theme.BORDER} !important;
    border-radius: 4px !important;
    color: {Theme.TEXT_SECONDARY} !important;
    pointer-events: auto !important;
    z-index: 999 !important;
    opacity: 1 !important;
    visibility: visible !important;
    display: flex !important;
}}

/* All sidebar text baseline */
section[data-testid="stSidebar"] * {{
    font-family: 'Inter', sans-serif !important;
}}

/* Scrollbars */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {Theme.BORDER_HOVER}; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: {Theme.TEXT_MUTED}; }}

/* Metric Cards - Ultra Sharp */
div[data-testid="stMetric"] {{
    background: {Theme.BG_CARD};
    border: 1px solid {Theme.BORDER};
    border-radius: 4px;
    padding: 20px 24px;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
    overflow: hidden;
}}
div[data-testid="stMetric"]::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {Theme.ACCENT_MAIN}, transparent);
    opacity: 0.4;
}}
div[data-testid="stMetric"]:hover {{
    border-color: {Theme.ACCENT_MAIN};
    background: {Theme.BG_HOVER};
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}}
div[data-testid="stMetric"] label, div[data-testid="stMetric"] label > div {{
    color: {Theme.TEXT_SECONDARY} !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    line-height: 1.4 !important;
}}
div[data-testid="stMetric"] [data-testid="stMetricValue"], div[data-testid="stMetric"] [data-testid="stMetricValue"] > div {{
    color: {Theme.TEXT_PRIMARY} !important;
    font-weight: 500 !important;
    font-size: 1.55rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.5px;
    margin-top: 8px;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    line-height: 1.2 !important;
}}

/* Typography Elements */
h1, h2, h3, h4, .stMarkdown p {{ color: {Theme.TEXT_PRIMARY}; }}

/* Base UI Components */
.glass-card {{
    background: {Theme.BG_CARD};
    border: 1px solid {Theme.BORDER};
    border-radius: 4px;
    padding: 24px;
    margin-bottom: 24px;
}}
.card-header-flex {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid {Theme.BORDER};
}}
.card-title {{
    color: {Theme.TEXT_PRIMARY};
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
.card-subtitle {{
    color: {Theme.TEXT_SECONDARY};
    font-size: 0.78rem;
    margin-top: 6px;
    line-height: 1.5;
}}

.info-banner {{
    background: {Theme.ACCENT_GLOW};
    border-left: 2px solid {Theme.ACCENT_MAIN};
    padding: 16px 24px;
    color: {Theme.TEXT_PRIMARY};
    font-size: 0.82rem;
    font-weight: 400;
    line-height: 1.6;
    margin-bottom: 32px;
    letter-spacing: 0.2px;
}}

/* Page Headers */
.page-title {{
    font-size: 1.6rem; font-weight: 600; letter-spacing: -0.02em; color: {Theme.TEXT_PRIMARY};
}}
.status-badge {{
    background: rgba(65,92,140,0.12);
    border: 1px solid {Theme.ACCENT_MAIN};
    color: {Theme.TEXT_PRIMARY};
    font-size: 0.65rem; font-weight: 600;
    padding: 6px 14px; border-radius: 3px;
    text-transform: uppercase; letter-spacing: 1.5px;
    font-family: 'JetBrains Mono', monospace;
}}
.status-indicator {{
    color: {Theme.RISK_LOW};
    margin-right: 6px;
}}

/* Controls — sidebar labels clearly visible */
.stSelectbox label, .stMultiSelect label, .stSlider label, .stNumberInput label, .stCaption {{
    font-size: 0.78rem !important;
    color: {Theme.TEXT_PRIMARY} !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
}}
.stSlider [data-testid="stThumbValue"] {{ color: {Theme.TEXT_PRIMARY} !important; font-weight: 600 !important; }}

/* Sidebar radio labels */
section[data-testid="stSidebar"] [data-testid="stRadio"] label p,
section[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    color: {Theme.TEXT_SECONDARY} !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}}
section[data-testid="stSidebar"] [aria-checked="true"] p {{
    color: {Theme.TEXT_PRIMARY} !important;
    font-weight: 600 !important;
}}

/* Multiselect tags */
[data-baseweb="tag"] {{ background: rgba(65,92,140,0.25) !important; }}
[data-baseweb="tag"] span {{ color: {Theme.TEXT_PRIMARY} !important; font-size: 0.78rem !important; }}

/* Sidebar specific */
.brand-title {{ font-size: 1.1rem; font-weight: 700; color: {Theme.TEXT_PRIMARY}; letter-spacing: 0.5px; }}
.brand-sub {{ font-size: 0.62rem; color: {Theme.ACCENT_MAIN}; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; margin-top: 6px; }}

/* st.caption override */
small, .stCaption p {{ color: {Theme.TEXT_SECONDARY} !important; font-size: 0.72rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 1px !important; }}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATA LAYER
# ==============================================================================
@st.cache_data(show_spinner="Evaluating proprietary models...")
def load_data() -> pd.DataFrame:
    """Robust data loader with explicit exception handling."""
    fp = os.path.join(os.path.dirname(__file__), "final_dataset.csv")
    if not os.path.exists(fp):
        logger.warning(f"Data file disconnected: {fp}")
        return pd.DataFrame()
    try:
        return pd.read_csv(fp)
    except Exception as e:
        logger.error(f"Ingestion fault: {str(e)}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.error("SYSTEM HALT: Core dataset (`final_dataset.csv`) disconnected. Verify upstream pipeline.")
    st.stop()

# Feature Mapping Dictionaries
FRIENDLY_NAMES = {
    "grade": "Credit Grade", "int_rate": "Nominal Interest", 
    "all_util": "Revolving Utilization", "max_bal_bc": "Max Card Exposure", 
    "mths_since_rcnt_il": "Installment Recency", "total_bal_il": "Installment Aggregate", 
    "il_util": "Installment Utilization", "target": "Observed Default", 
    "prob": "Modeled PD", "risk_bucket": "Risk Classification"
}
RISK_COLOR_MAP = {"Low": Theme.RISK_LOW, "Medium": Theme.RISK_MED, "High": Theme.RISK_HIGH}

def format_currency(val: float) -> str:
    """Financial-grade numerical formatter."""
    if abs(val) >= 1_000_000_000: return f"${val/1_000_000_000:.2f}B"
    if abs(val) >= 1_000_000: return f"${val/1_000_000:.2f}M"
    if abs(val) >= 1_000: return f"${val/1_000:.1f}K"
    return f"${val:,.0f}"

# ==============================================================================
# UI COMPONENTS
# ==============================================================================
def render_header(title: str, status: str = "SYS.ONLINE"):
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid {Theme.BORDER}; padding-bottom:16px; margin-bottom:28px;">
        <div class="page-title">{title}</div>
        <div class="status-badge"><span class="status-indicator">■</span> {status}</div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR LOGIC (Control Panel)
# ==============================================================================
with st.sidebar:
    st.markdown(f"""
    <div style="padding-bottom: 24px; border-bottom: 1px solid {Theme.BORDER}; margin-bottom: 24px;">
        <div class="brand-title">Lender's Club</div>
        <div class="brand-sub">Stratos Risk Engine v2</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("MODULE SELECT", ["Portfolio Overview", "Risk Vector Mapping", "Macro Stress Testing", "Model Diagnostics"], 
                    label_visibility="collapsed")
    
    st.markdown("---")
    st.caption("PORTFOLIO FILTERING")
    
    available_buckets = sorted(df["risk_bucket"].dropna().unique().tolist())
    if not available_buckets: available_buckets = ["Low", "Medium", "High"]
    
    selected_buckets = st.multiselect("Risk Exposure Tiers", available_buckets, default=available_buckets, 
                                      help="Filter sub-portfolios derived from the classification engine.")
    
    prob_min, prob_max = float(df["prob"].min()), float(df["prob"].max())
    prob_range = st.slider("PD Tolerance Constraint", prob_min, round(prob_max, 2), (prob_min, round(prob_max, 2)), format="%.3f")

    st.markdown("---")
    st.caption("MACROENVIRONMENT")
    stress_mult = st.slider("Systemic Shock (PD Multiplier)", 1.0, 3.0, 1.0, 0.1, 
                            help="Apply uniform beta expansion across all predicted probabilities.")
    lgd = st.number_input("Loss Given Default (LGD Matrix)", 0.0, 1.0, 0.60, 0.05, format="%.2f")
    ead_avg = st.number_input("Average EAD Anchor ($)", value=125_000, step=25_000)

# ==============================================================================
# CORE DATA PROCESSING & STATE
# ==============================================================================
mask = (df["risk_bucket"].isin(selected_buckets)) & (df["prob"] >= prob_range[0]) & (df["prob"] <= prob_range[1])
fdf = df[mask].copy()

# Base calculations
fdf["ECL"] = fdf["prob"] * lgd * ead_avg
fdf["stressed_prob"] = (fdf["prob"] * stress_mult).clip(upper=1.0)
fdf["stressed_ECL"] = fdf["stressed_prob"] * lgd * ead_avg

# Rigorous cut bins based on typical banking PDs (e.g., Prime, Near-Prime, Subprime)
bins = [-1.0, 0.08, 0.20, 2.0]
fdf["stressed_bucket"] = pd.cut(fdf["stressed_prob"], bins=bins, labels=["Low", "Medium", "High"], ordered=True)

# Pre-calculate main values
n_facilities = len(fdf)
n_defaults = fdf["target"].sum() if "target" in fdf else 0
def_rate = (n_defaults / n_facilities * 100) if n_facilities > 0 else 0
mean_pd = (fdf["prob"].mean() * 100) if n_facilities > 0 else 0
total_ecl = fdf["ECL"].sum() if n_facilities > 0 else 0
high_risk_flag = (fdf["risk_bucket"] == "High").sum()
high_pct = (high_risk_flag / n_facilities * 100) if n_facilities > 0 else 0

# ==============================================================================
# VIEW 1: PORTFOLIO OVERVIEW
# ==============================================================================
if page == "Portfolio Overview":
    render_header("Executive Portfolio Overview")
    
    st.markdown('<div class="info-banner">Aggregating real-time retail credit capital exposure. Metrics are dynamically driven by XGBoost predictive Probability of Default (PD) scoring models and Basel III framework approximations.</div>', unsafe_allow_html=True)
    
    # Advanced KPI block
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Facilities", f"{n_facilities:,}")
    c2.metric("Observed Defaults", f"{n_defaults:,}")
    c3.metric("Obs. Default Rate", f"{def_rate:.1f}%")
    c4.metric("Modeled Mean PD", f"{mean_pd:.2f}%")
    c5.metric("Net Expected Loss", format_currency(total_ecl))
    c6.metric("High Risk Concentration", f"{high_pct:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    
    r1c1, r1c2, r1c3 = st.columns(3)
    
    with r1c1:
        st.markdown('<div class="glass-card"><div class="card-header-flex"><div><div class="card-title">Exposure Density</div><div class="card-subtitle">Volume distribution by algorithmic risk tier</div></div></div>', unsafe_allow_html=True)
        counts = fdf["risk_bucket"].value_counts().reset_index()
        counts.columns = ["Tier", "Volume"]
        fig1 = px.pie(counts, names="Tier", values="Volume", color="Tier", color_discrete_map=RISK_COLOR_MAP, hole=0.74)
        fig1.update_layout(**Theme.get_plotly_layout(), height=260, showlegend=False)
        fig1.update_traces(textinfo="percent+label", textfont_size=11, textfont_color=Theme.TEXT_PRIMARY,
                           marker=dict(line=dict(color=Theme.BG_CARD, width=3)))
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r1c2:
        st.markdown('<div class="glass-card"><div class="card-header-flex"><div><div class="card-title">Realized Default Velocity</div><div class="card-subtitle">Ground-truth default occurrence per tier</div></div></div>', unsafe_allow_html=True)
        if "target" in fdf.columns:
            dr_df = fdf.groupby("risk_bucket")["target"].mean().reset_index()
            dr_df["Rate Analysis"] = dr_df["target"] * 100
            fig2 = px.bar(dr_df, x="risk_bucket", y="Rate Analysis", color="risk_bucket", color_discrete_map=RISK_COLOR_MAP, text_auto=".1f")
            fig2.update_layout(**Theme.get_plotly_layout(), height=260, showlegend=False, xaxis_title="", yaxis_title="Yield (%)")
            fig2.update_traces(textposition="outside", marker_line_width=0, opacity=0.9, width=0.5)
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with r1c3:
        st.markdown('<div class="glass-card"><div class="card-header-flex"><div><div class="card-title">Capital Provisioning Allocation</div><div class="card-subtitle">ECL requirement segmented by classifier</div></div></div>', unsafe_allow_html=True)
        el_df = fdf.groupby("risk_bucket")["ECL"].sum().reset_index()
        fig3 = px.bar(el_df, x="risk_bucket", y="ECL", color="risk_bucket", color_discrete_map=RISK_COLOR_MAP, text_auto="$.2s")
        fig3.update_layout(**Theme.get_plotly_layout(), height=260, showlegend=False, xaxis_title="", yaxis_title="ECL Liability ($)")
        fig3.update_traces(textposition="outside", marker_line_width=0, opacity=0.9, width=0.5)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c_b1, c_b2 = st.columns([1, 1])
    with c_b1:
        st.markdown('<div class="glass-card"><div class="card-header-flex"><div><div class="card-title">Continuous PD Density Function</div><div class="card-subtitle">High-resolution probability density mapped across active facilities</div></div></div>', unsafe_allow_html=True)
        fig4 = px.histogram(fdf, x="prob", nbins=80, color="risk_bucket", color_discrete_map=RISK_COLOR_MAP,
                            barmode="stack", opacity=0.85)
        fig4.update_layout(**Theme.get_plotly_layout(), height=360, xaxis_title="Modeled Probability of Default (PD)", yaxis_title="Volume Density")
        fig4.update_traces(marker_line_width=0)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c_b2:
        st.markdown('<div class="glass-card"><div class="card-header-flex"><div><div class="card-title">Risk-Pricing Efficiency Map</div><div class="card-subtitle">Interest Rate dispersion relative to algorithmic PD severity</div></div></div>', unsafe_allow_html=True)
        sample = fdf.sample(min(3000, len(fdf)), random_state=42)
        fig5 = px.scatter(sample, x="prob", y="int_rate", color="risk_bucket", color_discrete_map=RISK_COLOR_MAP,
                          opacity=0.45)
        fig5.update_layout(**Theme.get_plotly_layout(), height=360, xaxis_title="PD Anchor", yaxis_title="Nominal Yield")
        fig5.update_traces(marker=dict(size=4, line=dict(width=0)))
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# VIEW 2: RISK VECTOR MAPPING 
# ==============================================================================
elif page == "Risk Vector Mapping":
    render_header("Risk Vector Stratification", "ANALYTICS")
    
    st.markdown('<div class="info-banner">Examine how specific borrower dimensionality drives resultant capital risk categorization. Select an underlying engineered feature to analyze structural impacts across buckets.</div>', unsafe_allow_html=True)
    
    num_cols = fdf.select_dtypes(include=[np.number]).columns.tolist()
    valid_cols = [c for c in num_cols if c not in ["target", "prob", "ECL", "stressed_prob", "stressed_ECL"]]
    
    sel_feat = st.selectbox("Isolate Feature Vector", valid_cols, format_func=lambda x: FRIENDLY_NAMES.get(x, x))
    
    r1, r2 = st.columns([1, 1])
    with r1:
        st.markdown(f'<div class="glass-card"><div class="card-header-flex"><div><div class="card-title">Feature Probability Landscape</div><div class="card-subtitle">Density overlay for {FRIENDLY_NAMES.get(sel_feat, sel_feat)}</div></div></div>', unsafe_allow_html=True)
        fig = px.histogram(fdf, x=sel_feat, nbins=60, color="risk_bucket", color_discrete_map=RISK_COLOR_MAP,
                           barmode="overlay", opacity=0.5)
        fig.update_layout(**Theme.get_plotly_layout(), height=380, xaxis_title=FRIENDLY_NAMES.get(sel_feat, sel_feat))
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with r2:
        st.markdown(f'<div class="glass-card"><div class="card-header-flex"><div><div class="card-title">Interquartile Divergence</div><div class="card-subtitle">Box plot spread metrics stratified by severity</div></div></div>', unsafe_allow_html=True)
        fig2 = px.box(fdf, x="risk_bucket", y=sel_feat, color="risk_bucket", color_discrete_map=RISK_COLOR_MAP)
        fig2.update_layout(**Theme.get_plotly_layout(), height=380, showlegend=False, xaxis_title="", yaxis_title=FRIENDLY_NAMES.get(sel_feat, sel_feat))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# VIEW 3: MACRO STRESS TESTING
# ==============================================================================
elif page == "Macro Stress Testing":
    render_header("Macroeconomic Shock Simulator", "CCAR PROTOCOL")
    
    st.markdown('<div class="info-banner">Executing internal-ratings-based (IRB) stress routines. We impose uniform systemic degradation across the asset base to stress-test tier migration limits and capital provision deficiencies.</div>', unsafe_allow_html=True)

    if stress_mult == 1.0:
        st.info("System operating at baseline parameters. Modulate the 'Systemic Shock' slider in the control panel to execute an adverse scenario.")
    else:
        norm_total = fdf["ECL"].sum()
        str_total = fdf["stressed_ECL"].sum()
        delta = str_total - norm_total
        delta_pct = (delta / norm_total * 100) if norm_total > 0 else 0

        st.markdown(f"<h3 style='color:{Theme.TEXT_PRIMARY}; font-size: 1.05rem; font-weight: 500; margin-bottom: 24px;'>Parameter Shock: {stress_mult}x Contagion Event Modeled</h3>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Baseline Allocation", format_currency(norm_total))
        m2.metric("Stressed Obligation", format_currency(str_total), delta_color="inverse")
        m3.metric("Capital Shortfall Liability", format_currency(delta))
        m4.metric("Shock Magnitude Delta", f"+{delta_pct:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1.4])
        
        with col_left:
            st.markdown('<div class="glass-card"><div class="card-header-flex"><div><div class="card-title">Asset Transition Protocol</div><div class="card-subtitle">Facility migration vectors (%) into stress tiers</div></div></div>', unsafe_allow_html=True)
            if "stressed_bucket" in fdf.columns:
                migration = pd.crosstab(fdf["risk_bucket"], fdf["stressed_bucket"], 
                                      rownames=["Origin"], colnames=["Destination"], normalize='index') * 100
                st.dataframe(
                    migration,
                    column_config={
                        "Low": st.column_config.NumberColumn(format="%.1f%%"),
                        "Medium": st.column_config.NumberColumn(format="%.1f%%"),
                        "High": st.column_config.NumberColumn(format="%.1f%%"),
                    },
                    use_container_width=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_right:
            st.markdown('<div class="glass-card"><div class="card-header-flex"><div><div class="card-title">Liability Deformation Curve</div><div class="card-subtitle">Modeled vs Actual provisioning requirement</div></div></div>', unsafe_allow_html=True)
            ecl_norm = fdf.groupby("risk_bucket")["ECL"].sum().reset_index()
            ecl_norm.columns = ["Tier", "Liability"]
            ecl_norm["Model State"] = "Baseline"
            
            ecl_str = fdf.groupby("stressed_bucket")["stressed_ECL"].sum().dropna().reset_index()
            ecl_str.columns = ["Tier", "Liability"]
            ecl_str["Model State"] = f"Adverse Shock"
            
            combined = pd.concat([ecl_norm, ecl_str])
            fig_stress = px.bar(combined, x="Tier", y="Liability", color="Model State", barmode="group",
                                color_discrete_map={"Baseline": Theme.ACCENT_MAIN, f"Adverse Shock": Theme.RISK_MED})
            fig_stress.update_layout(**Theme.get_plotly_layout(), height=320)
            fig_stress.update_traces(marker_line_width=0, opacity=0.95)
            st.plotly_chart(fig_stress, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# VIEW 4: MODEL DIAGNOSTICS
# ==============================================================================
elif page == "Model Diagnostics":
    render_header("Classifier Integrity & Logic", "DIAGNOSTICS")
    
    st.markdown('<div class="info-banner">Direct mathematical introspection overlay. Utilizing pairwise collinearity extraction and absolute effect magnitudes to establish structural credibility of the core predictive logic layer.</div>', unsafe_allow_html=True)
    
    feat_subset = [c for c in ["int_rate", "all_util", "max_bal_bc", "mths_since_rcnt_il", "total_bal_il", "il_util", "prob"] if c in fdf.columns]
    
    c1, c2 = st.columns([1.2, 1])
    
    with c1:
        st.markdown('<div class="glass-card"><div class="card-header-flex"><div><div class="card-title">Multicollinearity Array</div><div class="card-subtitle">Pearson inter-feature correlation detection surface</div></div></div>', unsafe_allow_html=True)
        corr_matrix = fdf[feat_subset].corr()
        labels = [FRIENDLY_NAMES.get(c, c).split(" ")[0] for c in corr_matrix.columns]
        
        # Expert gradient mapping (Dark slate -> Muted Bronze -> White)
        colors = [[0, Theme.BG_CARD], [0.5, Theme.ACCENT_MAIN], [1, Theme.TEXT_PRIMARY]]
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values, x=labels, y=labels, colorscale=colors,
            text=np.round(corr_matrix.values, 2), texttemplate="%{text}",
            textfont={"size": 11, "color": Theme.TEXT_PRIMARY},
            hoverinfo="z"
        ))
        fig.update_layout(**Theme.get_plotly_layout(), height=450, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card"><div class="card-header-flex"><div><div class="card-title">Scalar Target Influence</div><div class="card-subtitle">Absolute correlation magnitude against binary default realization</div></div></div>', unsafe_allow_html=True)
        if "target" in fdf.columns:
            imp = fdf[feat_subset].corrwith(fdf["target"]).abs().sort_values(ascending=True).reset_index()
            imp.columns = ["Feature", "Magnitude"]
            imp["Name"] = imp["Feature"].map(FRIENDLY_NAMES)
            
            fig2 = px.bar(imp, x="Magnitude", y="Name", orientation="h")
            fig2.update_traces(marker_color=Theme.ACCENT_MAIN, marker_line_width=0, opacity=0.9,
                               text=np.round(imp["Magnitude"], 3), textposition="outside")
            fig2.update_layout(**Theme.get_plotly_layout(), height=420, yaxis_title="")
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TERMINAL CLEANUP
# ==============================================================================
st.markdown(f"""
<div style="text-align:center; padding-top:40px; margin-top:40px; border-top:1px solid {Theme.BORDER};">
    <p style="color:{Theme.TEXT_MUTED}; font-size:0.6rem; font-weight:600; letter-spacing:0.18em; text-transform:uppercase; font-family:'JetBrains Mono', monospace;">
    STRATOS RISK ENGINE // PROPRIETARY INSTITUTIONAL ARCHITECTURE // BUILD.V4.2
    </p>
</div>
""", unsafe_allow_html=True)
