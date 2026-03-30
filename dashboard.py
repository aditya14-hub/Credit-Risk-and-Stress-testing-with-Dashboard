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
    page_title="Stratos Analytics — Risk Core",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# THEME DEFINITIONS (Lead UI/UX Grade)
# ==============================================================================
class Theme:
    """Centralized theme configuration for consistent CSS and Plotly styling."""
    
    # Core Palette (Deep Slate & Muted Accents - Low eye strain, high contrast)
    BG_BASE = "#090B0F"          # Deepest slate blue
    BG_CARD = "#11141A"          # Elevated slate
    BG_HOVER = "#181C24"         # Interaction state
    
    TEXT_PRIMARY = "#E2E8F0"     # Crisp off-white
    TEXT_SECONDARY = "#94A3B8"   # Slate 400
    TEXT_MUTED = "#64748B"       # Slate 500
    
    BORDER = "#1E293B"           # Slate 800
    BORDER_HOVER = "#334155"     # Slate 700
    
    # Risk Indicators (Desaturated, professional)
    RISK_LOW = "#10B981"         # Muted Emerald
    RISK_MED = "#F59E0B"         # Muted Amber
    RISK_HIGH = "#EF4444"        # Muted Crimson
    
    # Accents
    ACCENT_MAIN = "#3B82F6"      # Institutional Blue
    ACCENT_ALT = "#6366F1"       # Indigo variant
    
    @classmethod
    def get_plotly_layout(cls) -> Dict[str, Any]:
        return dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color=cls.TEXT_SECONDARY, size=12),
            margin=dict(l=30, r=20, t=40, b=30),
            xaxis=dict(
                gridcolor=cls.BORDER,
                zerolinecolor=cls.BORDER,
                tickfont=dict(color=cls.TEXT_MUTED, size=11),
                title_font=dict(color=cls.TEXT_SECONDARY, size=12, weight="500"),
            ),
            yaxis=dict(
                gridcolor=cls.BORDER,
                zerolinecolor=cls.BORDER,
                tickfont=dict(color=cls.TEXT_MUTED, size=11),
                title_font=dict(color=cls.TEXT_SECONDARY, size=12, weight="500"),
            ),
            hoverlabel=dict(
                bgcolor=cls.BG_CARD,
                font_size=13,
                font_family="Inter",
                font_color=cls.TEXT_PRIMARY,
                bordercolor=cls.BORDER_HOVER
            ),
            legend=dict(
                font=dict(color=cls.TEXT_SECONDARY, size=12),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)"
            ),
            coloraxis_colorbar=dict(
                tickfont=dict(color=cls.TEXT_MUTED),
                titlefont=dict(color=cls.TEXT_SECONDARY)
            )
        )

# Inject Global CSS
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif !important;
}}

/* Base Theming */
.stApp {{ background: {Theme.BG_BASE} !important; }}
#MainMenu, footer, header, [data-testid="stHeaderActionElements"], .stAppDeployButton, [data-testid="manage-app-button"] {{ display: none !important; }}

/* Sidebar Customization */
section[data-testid="stSidebar"] {{
    background: {Theme.BG_CARD} !important;
    border-right: 1px solid {Theme.BORDER};
}}
[data-testid="collapsedControl"] {{
    color: {Theme.TEXT_SECONDARY} !important;
    background: {Theme.BG_CARD} !important;
    border: 1px solid {Theme.BORDER};
    border-radius: 4px;
}}

/* Scrollbars */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {Theme.BORDER}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {Theme.BORDER_HOVER}; }}

/* Metric Cards */
div[data-testid="stMetric"] {{
    background: {Theme.BG_CARD};
    border: 1px solid {Theme.BORDER};
    border-radius: 8px;
    padding: 20px;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}}
div[data-testid="stMetric"]:hover {{
    border-color: {Theme.ACCENT_MAIN};
    background: {Theme.BG_HOVER};
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}}
div[data-testid="stMetric"] label {{
    color: {Theme.TEXT_MUTED} !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: {Theme.TEXT_PRIMARY} !important;
    font-weight: 600 !important;
    font-size: 1.5rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.02em;
    margin-top: 4px;
}}

/* Typography Elements */
h1, h2, h3, h4, .stMarkdown p {{ color: {Theme.TEXT_PRIMARY}; }}

/* Base UI Components (Cards, Banners) */
.glass-card {{
    background: {Theme.BG_CARD};
    border: 1px solid {Theme.BORDER};
    border-radius: 8px;
    padding: 24px;
    margin-bottom: 20px;
}}
.card-title {{
    color: {Theme.TEXT_PRIMARY};
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 6px;
    display: flex; align-items: center; gap: 8px;
}}
.card-title::before {{
    content: ''; width: 4px; height: 16px;
    background: {Theme.ACCENT_MAIN}; border-radius: 2px;
}}
.card-subtitle {{
    color: {Theme.TEXT_SECONDARY};
    font-size: 0.75rem;
    margin-bottom: 16px;
    padding-left: 12px;
}}

.info-banner {{
    background: rgba(59, 130, 246, 0.05);
    border-left: 3px solid {Theme.ACCENT_MAIN};
    padding: 16px 20px;
    border-radius: 4px;
    color: {Theme.TEXT_SECONDARY};
    font-size: 0.85rem;
    line-height: 1.6;
    margin-bottom: 24px;
}}

/* Page Headers */
.page-title {{
    font-size: 1.5rem; font-weight: 600; letter-spacing: -0.02em; color: {Theme.TEXT_PRIMARY};
}}
.status-badge {{
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.2);
    color: {Theme.RISK_LOW};
    font-size: 0.65rem; font-weight: 600;
    padding: 4px 10px; border-radius: 4px;
    text-transform: uppercase; letter-spacing: 0.05em;
}}

/* Controls */
.stSelectbox label, .stMultiSelect label, .stSlider label {{
    font-size: 0.8rem !important; color: {Theme.TEXT_SECONDARY} !important; font-weight: 500 !important;
}}

/* Sidebar specific */
.brand-title {{ font-size: 1.1rem; font-weight: 700; color: {Theme.TEXT_PRIMARY}; }}
.brand-sub {{ font-size: 0.65rem; color: {Theme.TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.1em; }}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATA LAYER
# ==============================================================================
@st.cache_data(show_spinner="Loading portfolio intelligence...")
def load_data() -> pd.DataFrame:
    """Robust data loader with explicit exception handling."""
    fp = os.path.join(os.path.dirname(__file__), "final_dataset.csv")
    if not os.path.exists(fp):
        logger.warning(f"Data file not found at {fp}")
        return pd.DataFrame()
    try:
        return pd.read_csv(fp)
    except Exception as e:
        logger.error(f"Error reading dataset: {str(e)}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.error("⚠️ Primary dataset (`final_dataset.csv`) unavailable. Ensure the data pipeline has executed successfully.")
    st.stop()

# Feature mapping dictionaries
FRIENDLY_NAMES = {
    "grade": "Credit Grade", "int_rate": "Effective Interest Rate", 
    "all_util": "Credit Utilization Ratio", "max_bal_bc": "Max Revolving Balance", 
    "mths_since_rcnt_il": "Installment Recency (Months)", "total_bal_il": "Aggregate Installment Balance", 
    "il_util": "Installment Utilization", "target": "Observed Default", 
    "prob": "Predicted PD", "risk_bucket": "Risk Classification"
}
RISK_COLOR_MAP = {"Low": Theme.RISK_LOW, "Medium": Theme.RISK_MED, "High": Theme.RISK_HIGH}

# Formatter
def format_currency(val: float) -> str:
    """Financia-grade formatter handles robust scaling."""
    if abs(val) >= 1_000_000_000: return f"${val/1_000_000_000:.2f}B"
    if abs(val) >= 1_000_000: return f"${val/1_000_000:.2f}M"
    if abs(val) >= 1_000: return f"${val/1_000:.1f}K"
    return f"${val:,.0f}"

# ==============================================================================
# UI COMPONENTS
# ==============================================================================
def render_header(title: str, status: str = "SYSTEM ONLINE"):
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid {Theme.BORDER}; padding-bottom:16px; margin-bottom:24px;">
        <div class="page-title">{title}</div>
        <div class="status-badge">● {status}</div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR LOGIC
# ==============================================================================
with st.sidebar:
    st.markdown(f"""
    <div style="padding-bottom: 24px; border-bottom: 1px solid {Theme.BORDER}; margin-bottom: 24px;">
        <div class="brand-title">Stratos Analytics</div>
        <div class="brand-sub">Institutional Risk Engine</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("MODULE SELECT", ["Portfolio Overview", "Risk Segmentation", "Macro Stress Testing", "Model Diagnostics"], 
                    label_visibility="collapsed")
    
    st.markdown("---")
    st.caption("PORTFOLIO FILTERS")
    
    # Robust bucket extraction
    available_buckets = sorted(df["risk_bucket"].dropna().unique().tolist())
    if not available_buckets: available_buckets = ["Low", "Medium", "High"]
    
    selected_buckets = st.multiselect("Risk Exposure Tiers", available_buckets, default=available_buckets, 
                                      help="Select risk tiers derived from the XGBoost classification model.")
    
    prob_min, prob_max = float(df["prob"].min()), float(df["prob"].max())
    prob_range = st.slider("PD Tolerance Boundary", prob_min, prob_max, (prob_min, prob_max), format="%.3f",
                           help="Filter facilities based on Predicted Probability of Default (PD).")

    st.markdown("---")
    st.caption("MACRO SCENARIO")
    stress_mult = st.slider("PD Stress Shock Map (x)", 1.0, 3.0, 1.0, 0.1, 
                            help="Apply a multiplicative shock to baseline PDs to simulate economic downturns.")
    lgd = st.number_input("Loss Given Default (LGD)", 0.0, 1.0, 0.60, 0.05, format="%.2f")
    ead_avg = st.number_input("Facility EAD Anchor ($)", value=125_000, step=25_000)

# ==============================================================================
# CORE DATA PROCESSING
# ==============================================================================
# Apply filters
mask = (df["risk_bucket"].isin(selected_buckets)) & (df["prob"] >= prob_range[0]) & (df["prob"] <= prob_range[1])
filtered_df = df[mask].copy()

# Derived Financial Metrics
filtered_df["ECL"] = filtered_df["prob"] * lgd * ead_avg
filtered_df["stressed_prob"] = (filtered_df["prob"] * stress_mult).clip(upper=1.0)
filtered_df["stressed_ECL"] = filtered_df["stressed_prob"] * lgd * ead_avg

# Dynamic buckling for stressed scenarios based on quantiles or fixed logic
# Realistic boundaries for Low/Medium/High in retail credit
bins = [-np.inf, 0.10, 0.25, np.inf]
filtered_df["stressed_bucket"] = pd.cut(filtered_df["stressed_prob"], bins=bins, labels=["Low", "Medium", "High"])

# ==============================================================================
# VIEW: PORTFOLIO OVERVIEW
# ==============================================================================
if page == "Portfolio Overview":
    render_header("Portfolio Executive Overview")
    
    st.markdown('<div class="info-banner">Real-time aggregate view of retail credit exposures. Metrics utilize active predictive PD (Probability of Default) models and align with Basel III internal-ratings-based (IRB) approaches.</div>', unsafe_allow_html=True)
    
    # Safe aggregations
    n_facilities = len(filtered_df)
    n_defaults = filtered_df["target"].sum() if "target" in filtered_df else 0
    def_rate = (n_defaults / n_facilities * 100) if n_facilities > 0 else 0
    mean_pd = (filtered_df["prob"].mean() * 100) if n_facilities > 0 else 0
    total_ecl = filtered_df["ECL"].sum() if n_facilities > 0 else 0
    high_risk_flag = (filtered_df["risk_bucket"] == "High").sum()
    high_pct = (high_risk_flag / n_facilities * 100) if n_facilities > 0 else 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Active Facilities", f"{n_facilities:,}")
    col2.metric("Observed Defaults", f"{n_defaults:,}")
    col3.metric("Act. Default Rate", f"{def_rate:.1f}%")
    col4.metric("Avg Predicted PD", f"{mean_pd:.2f}%")
    col5.metric("Total ECL Model", format_currency(total_ecl))
    col6.metric("High Risk Concentration", f"{high_pct:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Upper Charts
    c_l, c_m, c_r = st.columns(3)
    
    with c_l:
        st.markdown('<div class="glass-card"><div class="card-title">Exposure Density</div><div class="card-subtitle">Facility volume by risk tier</div>', unsafe_allow_html=True)
        counts = filtered_df["risk_bucket"].value_counts().reset_index()
        counts.columns = ["Tier", "Volume"]
        fig1 = px.pie(counts, names="Tier", values="Volume", color="Tier", color_discrete_map=RISK_COLOR_MAP, hole=0.7)
        fig1.update_layout(**Theme.get_plotly_layout(), height=260, showlegend=True,
                           legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"))
        fig1.update_traces(textinfo="percent", textfont_size=12, textfont_color=Theme.TEXT_PRIMARY,
                           marker=dict(line=dict(color=Theme.BG_CARD, width=3)))
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_m:
        st.markdown('<div class="glass-card"><div class="card-title">Default Yields</div><div class="card-subtitle">Realized default percentage by tier</div>', unsafe_allow_html=True)
        if "target" in filtered_df.columns:
            dr_df = filtered_df.groupby("risk_bucket")["target"].mean().reset_index()
            dr_df["Rate Analysis"] = dr_df["target"] * 100
            fig2 = px.bar(dr_df, x="risk_bucket", y="Rate Analysis", color="risk_bucket", color_discrete_map=RISK_COLOR_MAP, text_auto=".1f")
            fig2.update_layout(**Theme.get_plotly_layout(), height=260, showlegend=False, xaxis_title="", yaxis_title="Rate (%)")
            fig2.update_traces(textposition="outside", marker_line_width=0, opacity=0.9)
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c_r:
        st.markdown('<div class="glass-card"><div class="card-title">Capital Provisioning</div><div class="card-subtitle">Expected Credit Loss (ECL) totals</div>', unsafe_allow_html=True)
        el_df = filtered_df.groupby("risk_bucket")["ECL"].sum().reset_index()
        fig3 = px.bar(el_df, x="risk_bucket", y="ECL", color="risk_bucket", color_discrete_map=RISK_COLOR_MAP, text_auto="$.2s")
        fig3.update_layout(**Theme.get_plotly_layout(), height=260, showlegend=False, xaxis_title="", yaxis_title="ECL Val")
        fig3.update_traces(textposition="outside", marker_line_width=0, opacity=0.9)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Lower Charts
    c_b1, c_b2 = st.columns([1.5, 1])
    with c_b1:
        st.markdown('<div class="glass-card"><div class="card-title">PD Probability Density Curve</div><div class="card-subtitle">Granular continuous distribution of internal PD predictions</div>', unsafe_allow_html=True)
        fig4 = px.histogram(filtered_df, x="prob", nbins=75, color="risk_bucket", color_discrete_map=RISK_COLOR_MAP,
                            barmode="stack", opacity=0.85, marginal="box")
        fig4.update_layout(**Theme.get_plotly_layout(), height=360, xaxis_title="Predicted Probability of Default (PD)", yaxis_title="Facility Count")
        fig4.update_traces(marker_line_width=0)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c_b2:
        st.markdown('<div class="glass-card"><div class="card-title">Risk Adjusted Pricing</div><div class="card-subtitle">Interest Rate calibration against PD Score</div>', unsafe_allow_html=True)
        sample = filtered_df.sample(min(3000, len(filtered_df)), random_state=42)
        fig5 = px.scatter(sample, x="prob", y="int_rate", color="risk_bucket", color_discrete_map=RISK_COLOR_MAP,
                          opacity=0.5, trendline="ols", trendline_color_override=Theme.TEXT_PRIMARY)
        fig5.update_layout(**Theme.get_plotly_layout(), height=360, xaxis_title="PD Score", yaxis_title="Int. Rate")
        fig5.update_traces(marker=dict(size=5, line=dict(width=0)))
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# VIEW: RISK SEGMENTATION
# ==============================================================================
elif page == "Risk Segmentation":
    render_header("Risk Characteristics Matrix")
    
    st.markdown('<div class="info-banner">Deep-dive multidimensional analysis mapping borrower characteristics directly to resultant capital risk tiers. Enables precise identification of correlative risk markers.</div>', unsafe_allow_html=True)
    
    num_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    valid_cols = [c for c in num_cols if c not in ["target", "prob", "ECL", "stressed_prob", "stressed_ECL"]]
    
    sel_feat = st.selectbox("Analyzed Variable Focus", valid_cols, format_func=lambda x: FRIENDLY_NAMES.get(x, x))
    
    r1, r2 = st.columns([1, 1])
    with r1:
        st.markdown(f'<div class="glass-card"><div class="card-title">Layered Density</div><div class="card-subtitle">Overlay progression across {FRIENDLY_NAMES.get(sel_feat, sel_feat)}</div>', unsafe_allow_html=True)
        fig = px.histogram(filtered_df, x=sel_feat, nbins=50, color="risk_bucket", color_discrete_map=RISK_COLOR_MAP,
                           barmode="overlay", opacity=0.6)
        fig.update_layout(**Theme.get_plotly_layout(), height=350, xaxis_title=FRIENDLY_NAMES.get(sel_feat, sel_feat))
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with r2:
        st.markdown(f'<div class="glass-card"><div class="card-title">Quantile Volatility</div><div class="card-subtitle">Box plot spread metrics by tier</div>', unsafe_allow_html=True)
        fig2 = px.box(filtered_df, x="risk_bucket", y=sel_feat, color="risk_bucket", color_discrete_map=RISK_COLOR_MAP)
        fig2.update_layout(**Theme.get_plotly_layout(), height=350, showlegend=False, xaxis_title="", yaxis_title=FRIENDLY_NAMES.get(sel_feat, sel_feat))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# VIEW: STRESS TESTING
# ==============================================================================
elif page == "Macro Stress Testing":
    render_header("Adverse Scenario Simulator", "SIMULATION MODE")
    
    st.markdown('<div class="info-banner">Utilizing CCAR/DFAST stylized methods. We apply uniform scaling factors to individual baseline PDs, forcing systemic downgrades, to determine gross capital adequacy shortfalls.</div>', unsafe_allow_html=True)

    if stress_mult == 1.0:
        st.info("Baseline models currently active. Adjust 'PD Stress Shock Map' in the sidebar parameters to invoke an economic shock.")
    else:
        norm_total = filtered_df["ECL"].sum()
        str_total = filtered_df["stressed_ECL"].sum()
        delta = str_total - norm_total
        delta_pct = (delta / norm_total * 100) if norm_total > 0 else 0

        st.markdown(f"<h3 style='color:{Theme.TEXT_PRIMARY}; font-size: 1.1rem; margin-bottom: 20px;'>Shock Protocol: {stress_mult}x Systemic Multiplier</h3>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Unstressed Provisioning", format_currency(norm_total))
        m2.metric("Stressed Provision Req.", format_currency(str_total), delta_color="inverse")
        m3.metric("Required Capital Injection", format_currency(delta))
        m4.metric("Capital Delta (%)", f"+{delta_pct:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.markdown('<div class="glass-card"><div class="card-title">Tier Migration Map</div><div class="card-subtitle">Borrower degradation paths</div>', unsafe_allow_html=True)
            # Create a clean migration matrix
            if "stressed_bucket" in filtered_df.columns:
                migration = pd.crosstab(filtered_df["risk_bucket"], filtered_df["stressed_bucket"], 
                                      rownames=["Baseline"], colnames=["Stressed"], normalize='index') * 100
                st.dataframe(migration.style.format("{:.1f}%").background_gradient(cmap="Blues", axis=None), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_right:
            st.markdown('<div class="glass-card"><div class="card-title">Provision Equivalence Shift</div><div class="card-subtitle">ECL variance between baseline operations and modeled adverse conditions</div>', unsafe_allow_html=True)
            
            ecl_norm = filtered_df.groupby("risk_bucket")["ECL"].sum().reset_index()
            ecl_norm.columns = ["Tier", "Capital Required"]
            ecl_norm["Scenario"] = "Base (Current)"
            
            ecl_str = filtered_df.groupby("stressed_bucket")["stressed_ECL"].sum().dropna().reset_index()
            ecl_str.columns = ["Tier", "Capital Required"]
            ecl_str["Scenario"] = f"Stressed (x{stress_mult})"
            
            combined = pd.concat([ecl_norm, ecl_str])
            fig_stress = px.bar(combined, x="Tier", y="Capital Required", color="Scenario", barmode="group",
                                color_discrete_map={"Base (Current)": Theme.ACCENT_ALT, f"Stressed (x{stress_mult})": Theme.RISK_HIGH})
            fig_stress.update_layout(**Theme.get_plotly_layout(), height=300)
            st.plotly_chart(fig_stress, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# VIEW: MODEL DIAGNOSTICS
# ==============================================================================
elif page == "Model Diagnostics":
    render_header("Predictive Logic Diagnostics", "XGBOOST CORE")
    
    st.markdown('<div class="info-banner">Quantitative interpretability architecture. Auditing internal feature collinearity protocols and feature importance vectors relative to observed defaults.</div>', unsafe_allow_html=True)
    
    feat_subset = [c for c in ["int_rate", "all_util", "max_bal_bc", "mths_since_rcnt_il", "total_bal_il", "il_util", "prob"] if c in filtered_df.columns]
    
    c1, c2 = st.columns([1.2, 1])
    
    with c1:
        st.markdown('<div class="glass-card"><div class="card-title">Pearson Collinearity Matrix</div><div class="card-subtitle">Detecting inter-feature noise mechanisms</div>', unsafe_allow_html=True)
        corr_matrix = filtered_df[feat_subset].corr()
        labels = [FRIENDLY_NAMES.get(c, c).split(" ")[0] for c in corr_matrix.columns]
        
        # Super clean, dark heatmap mapping
        colors = [[0, Theme.BG_BASE], [0.5, Theme.BORDER], [1, Theme.ACCENT_MAIN]]
        
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
        st.markdown('<div class="glass-card"><div class="card-title">Absolute Delta Importance</div><div class="card-subtitle">Correlation magnitude against realization metric</div>', unsafe_allow_html=True)
        if "target" in filtered_df.columns:
            imp = filtered_df[feat_subset].corrwith(filtered_df["target"]).abs().sort_values(ascending=True).reset_index()
            imp.columns = ["Feature", "Magnitude"]
            imp["Name"] = imp["Feature"].map(FRIENDLY_NAMES)
            
            fig2 = px.bar(imp, x="Magnitude", y="Name", orientation="h")
            fig2.update_traces(marker_color=Theme.ACCENT_ALT, marker_line_width=0, opacity=0.9,
                               text=np.round(imp["Magnitude"], 3), textposition="outside")
            fig2.update_layout(**Theme.get_plotly_layout(), height=430, yaxis_title="")
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TERMINATION & CLEANUP
# ==============================================================================
st.markdown(f"""
<div style="text-align:center; padding-top:40px; margin-top:40px; border-top:1px solid {Theme.BORDER};">
    <p style="color:{Theme.TEXT_MUTED}; font-size:0.65rem; font-weight:600; letter-spacing:0.15em; text-transform:uppercase;">
    STRATOS ANALYTICS &copy; 2026 // PROPRIETARY INSTITUTIONAL LOGIC
    </p>
</div>
""", unsafe_allow_html=True)
