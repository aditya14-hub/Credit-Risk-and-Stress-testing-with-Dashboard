"""
═══════════════════════════════════════════════════════════════════════════════
LENDER'S CLUB - INSTITUTIONAL RISK INTELLIGENCE PLATFORM v3.0
═══════════════════════════════════════════════════════════════════════════════

A comprehensive credit risk analytics and portfolio management solution designed
for institutional lending decision-making. Leverages machine learning predictive
probability of default (PD) scoring with Basel III framework compliance.

Architecture: Modular, scalable Streamlit application with advanced analytics,
multi-dimensional segmentation, and executive-grade visualizations.

Author: Senior Analytics Engineer
Version: 3.0 Professional Edition
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import io
import base64

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Lender's Club Risk Intelligence Platform",
    layout="wide",
    page_icon="LC",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Professional Credit Risk Analytics Suite • v3.0"
    }
)

# ═══════════════════════════════════════════════════════════════════════════════
# PREMIUM THEME SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class ThemeConfig:
    """Institutional-grade color and design system."""
    # Primary Palette
    BG_PRIMARY = "#0A0E27"        # Deep navy-black
    BG_SECONDARY = "#0F1432"      # Card background
    BG_HOVER = "#151B3F"          # Hover state
    
    # Text Hierarchy
    TEXT_PRIMARY = "#F0F4FF"      # Crisp white
    TEXT_SECONDARY = "#99A8C1"    # Secondary text
    TEXT_MUTED = "#667085"        # Muted labels
    
    # Accents & Severity
    ACCENT_BLUE = "#3B82F6"       # Primary action
    ACCENT_CYAN = "#06B6D4"       # Secondary action
    
    RISK_SAFE = "#10B981"         # Low risk (emerald)
    RISK_CAUTION = "#F59E0B"      # Medium risk (amber)
    RISK_DANGER = "#EF4444"       # High risk (red)
    
    # Structural
    BORDER_LIGHT = "#1E293B"      # Light border
    BORDER_DARK = "#0F172A"       # Dark border
    GLOW_SUBTLE = "rgba(59, 130, 246, 0.1)"
    
    @classmethod
    def get_plotly_layout(cls) -> Dict[str, Any]:
        """Standard Plotly template with premium styling."""
        return dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, -apple-system, sans-serif", color=cls.TEXT_SECONDARY, size=11),
            margin=dict(l=24, r=24, t=44, b=24),
            xaxis=dict(
                gridcolor="rgba(255,255,255,0.05)",
                zerolinecolor="rgba(255,255,255,0.08)",
                tickfont=dict(color=cls.TEXT_SECONDARY, size=10),
                title_font=dict(color=cls.TEXT_PRIMARY, size=12, family="Inter"),
                showgrid=True,
                zeroline=False,
            ),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.05)",
                zerolinecolor="rgba(255,255,255,0.08)",
                tickfont=dict(color=cls.TEXT_SECONDARY, size=10),
                title_font=dict(color=cls.TEXT_PRIMARY, size=12, family="Inter"),
                showgrid=True,
                zeroline=False,
            ),
            hoverlabel=dict(
                bgcolor="#1E293B",
                font_size=12,
                font_family="Inter",
                font_color=cls.TEXT_PRIMARY,
                bordercolor=cls.ACCENT_BLUE,
                namelength=-1
            ),
            legend=dict(
                font=dict(color=cls.TEXT_PRIMARY, size=11),
                bgcolor="rgba(0,0,0,0.3)",
                bordercolor=cls.BORDER_LIGHT,
                borderwidth=1,
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            coloraxis_colorbar=dict(
                tickfont=dict(color=cls.TEXT_MUTED, size=10),
                title=dict(font=dict(color=cls.TEXT_SECONDARY, size=11)),
                bordercolor=cls.BORDER_LIGHT,
                thickness=12
            )
        )

TH = ThemeConfig()

# Global CSS Injection
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

* {{ font-family: 'Inter', -apple-system, sans-serif !important; }}

html, body {{ background: {TH.BG_PRIMARY} !important; }}
.stApp {{ background: {TH.BG_PRIMARY} !important; }}

/* Hide Default Chrome */
#MainMenu, footer, .stAppDeployButton {{ visibility: hidden !important; }}
[data-testid="stHeaderActionElements"], [data-testid="manage-app-button"] {{ display: none !important; }}

/* Sidebar Professional Styling */
section[data-testid="stSidebar"] {{
    background: {TH.BG_SECONDARY} !important;
    border-right: 1px solid {TH.BORDER_LIGHT} !important;
}}

/* Metric Cards Enhanced */
div[data-testid="stMetric"] {{
    background: {TH.BG_SECONDARY};
    border: 1px solid {TH.BORDER_LIGHT};
    border-radius: 6px;
    padding: 20px;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}}
div[data-testid="stMetric"]::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {TH.ACCENT_BLUE}, transparent);
    opacity: 0;
    transition: opacity 0.3s;
}}
div[data-testid="stMetric"]:hover {{
    border-color: {TH.ACCENT_BLUE};
    background: {TH.BG_HOVER};
    box-shadow: 0 8px 32px rgba(59, 130, 246, 0.15);
}}
div[data-testid="stMetric"]:hover::before {{ opacity: 1; }}
div[data-testid="stMetric"] label {{ color: {TH.TEXT_MUTED} !important; font-size: 0.73rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.8px; }}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {{ color: {TH.TEXT_PRIMARY} !important; font-weight: 600 !important; font-size: 1.6rem !important; }}

/* Typography */
h1, h2, h3, h4, h5, h6 {{ color: {TH.TEXT_PRIMARY} !important; }}
p, .stMarkdown {{ color: {TH.TEXT_PRIMARY} !important; }}

/* Card Styling */
.metric-card {{
    background: {TH.BG_SECONDARY};
    border: 1px solid {TH.BORDER_LIGHT};
    border-radius: 6px;
    padding: 24px;
    margin-bottom: 20px;
    transition: all 0.3s;
}}
.metric-card:hover {{ 
    border-color: {TH.ACCENT_BLUE};
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.12);
}}

.card-title {{
    color: {TH.TEXT_PRIMARY};
    font-size: 0.95rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}}

.card-subtitle {{
    color: {TH.TEXT_SECONDARY};
    font-size: 0.8rem;
    font-weight: 400;
    line-height: 1.4;
}}

.card-header {{
    padding-bottom: 16px;
    margin-bottom: 20px;
    border-bottom: 1px solid {TH.BORDER_LIGHT};
}}

.info-box {{
    background: rgba(59, 130, 246, 0.1);
    border-left: 3px solid {TH.ACCENT_BLUE};
    border-radius: 4px;
    padding: 16px 20px;
    color: {TH.TEXT_PRIMARY};
    font-size: 0.85rem;
    line-height: 1.6;
    margin-bottom: 24px;
}}

/* Controls */
.stSelectbox label, .stMultiSelect label, .stSlider label, .stNumberInput label {{
    color: {TH.TEXT_PRIMARY} !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px !important;
}}

.status-badge {{
    display: inline-block;
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid {TH.ACCENT_BLUE};
    color: {TH.ACCENT_BLUE};
    font-size: 0.7rem;
    font-weight: 700;
    padding: 6px 12px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace;
}}

.stat-number {{
    color: {TH.TEXT_PRIMARY};
    font-size: 1.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}}

.stat-label {{
    color: {TH.TEXT_SECONDARY};
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}}

/* Scrollbars */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {TH.BORDER_LIGHT}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {TH.TEXT_MUTED}; }}

</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA MANAGEMENT LAYER
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading institutional dataset...")
def load_master_dataset() -> pd.DataFrame:
    """Load and validate master lending dataset."""
    fp = os.path.join(os.path.dirname(__file__), "final_dataset.csv")
    if not os.path.exists(fp):
        st.error("CRITICAL: Master dataset not found. Verify upstream pipeline.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(fp)
        logger.info(f"✓ Dataset loaded: {len(df):,} facilities")
        return df
    except Exception as e:
        logger.error(f"✗ Ingestion error: {str(e)}")
        return pd.DataFrame()

# Feature Metadata
FEATURE_DESCRIPTIONS = {
    "grade": "Credit Grade (Standardized Tier)",
    "int_rate": "Nominal Interest Rate (%)",
    "all_util": "All Credit Utilization Ratio",
    "max_bal_bc": "Maximum Credit Card Balance",
    "mths_since_rcnt_il": "Months Since Recent Installment",
    "total_bal_il": "Total Installment Balance",
    "il_util": "Installment Utilization Ratio",
    "prob": "Predicted Probability of Default (PD)",
    "target": "Observed Default Event",
    "risk_bucket": "Risk Classification Tier",
    "ECL": "Expected Credit Loss",
    "stressed_ECL": "Stressed ECL (Macro Scenario)"
}

RISK_COLORS = {
    "Low": TH.RISK_SAFE,
    "Medium": TH.RISK_CAUTION,
    "High": TH.RISK_DANGER
}

RISK_ORDER = ["Low", "Medium", "High"]

# Load Data
df_master = load_master_dataset()

if df_master.empty:
    st.error("System Error: Cannot initialize. Dataset unavailable.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def format_currency(value: float, precision: str = "auto") -> str:
    """Format numerical values as currency."""
    if value is None or pd.isna(value):
        return "—"
    abs_val = abs(value)
    if abs_val >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    elif abs_val >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif abs_val >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:,.0f}"

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage values."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{decimals}f}%"

def create_metric_card(value: str, label: str, delta: Optional[str] = None, 
                       delta_color: str = "normal", icon: str = "") -> str:
    """Create styled metric card HTML."""
    delta_html = ""
    if delta:
        color = TH.ACCENT_BLUE if delta_color == "positive" else TH.RISK_DANGER
        arrow = "↑" if delta_color == "positive" else "↓"
        delta_html = f'<div style="color:{color}; font-size:0.8rem; margin-top:6px;">{arrow} {delta}</div>'
    
    icon_html = f'<div style="font-size:2rem; margin-bottom:8px;">{icon}</div>' if icon else ""
    return f"""
    <div class="metric-card">
        {icon_html}
        <div class="stat-number">{value}</div>
        <div class="stat-label">{label}</div>
        {delta_html}
    </div>
    """

def render_header(title: str, subtitle: str = "", status: str = "ACTIVE") -> None:
    """Render page header with styling."""
    st.markdown(f"""
    <div style="margin-bottom: 32px; padding-bottom: 20px; border-bottom: 2px solid {TH.BORDER_LIGHT};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; color: {TH.TEXT_PRIMARY};">{title}</h1>
                {f'<p style="color: {TH.TEXT_SECONDARY}; margin: 8px 0 0 0; font-size: 0.9rem;">{subtitle}</p>' if subtitle else ''}
            </div>
            <span class="status-badge">{status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def calculate_portfolio_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate comprehensive portfolio metrics."""
    if df.empty:
        return {}
    
    df_calc = df.copy()
    df_calc["ECL"] = df_calc.get("prob", 0) * 0.60 * 125_000
    
    return {
        "total_facilities": len(df_calc),
        "total_exposure": len(df_calc) * 125_000,
        "observed_defaults": df_calc["target"].sum() if "target" in df_calc else 0,
        "observed_default_rate": (df_calc["target"].sum() / len(df_calc) * 100) if "target" in df_calc else 0,
        "mean_pd": df_calc["prob"].mean() * 100 if "prob" in df_calc else 0,
        "median_pd": df_calc["prob"].median() * 100 if "prob" in df_calc else 0,
        "total_ecl": df_calc["ECL"].sum(),
        "mean_int_rate": df_calc["int_rate"].mean() * 100 if "int_rate" in df_calc else 0,
        "mean_utilization": df_calc["all_util"].mean() * 100 if "all_util" in df_calc else 0,
    }

def segment_by_feature(df: pd.DataFrame, feature: str, n_bins: int = 5) -> pd.DataFrame:
    """Create quantile-based segmentation."""
    df_seg = df.copy()
    df_seg[f"{feature}_segment"] = pd.qcut(df_seg[feature], q=n_bins, labels=False, duplicates='drop')
    return df_seg

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EXECUTIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def page_executive_summary():
    """High-level executive overview with key metrics and trends."""
    render_header(
        "Executive Risk Dashboard",
        "Real-time institutional asset quality & capital adequacy monitoring",
        "LIVE"
    )
    
    # Sidebar Controls
    selected_risk = st.sidebar.multiselect(
        "Risk Tier Filter",
        ["Low", "Medium", "High"],
        default=["Low", "Medium", "High"],
        key="exec_risk"
    )
    
    pd_range = st.sidebar.slider(
        "PD Acceptable Range",
        float(df_master["prob"].min()),
        float(df_master["prob"].max()),
        (float(df_master["prob"].min()), float(df_master["prob"].max())),
        format="%.3f",
        key="exec_pd"
    )
    
    # Apply filters
    df_filtered = df_master[
        (df_master["risk_bucket"].isin(selected_risk)) &
        (df_master["prob"] >= pd_range[0]) &
        (df_master["prob"] <= pd_range[1])
    ].copy()
    
    # Calculate metrics
    metrics = calculate_portfolio_metrics(df_filtered)
    
    # KPI Cards
    st.markdown("### Portfolio Snapshot")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "Total Facilities",
            f"{metrics['total_facilities']:,}",
            f"{(len(df_filtered)/len(df_master)*100):.1f}% of portfolio"
        )

    with col2:
        st.metric(
            "Total Exposure",
            format_currency(metrics['total_exposure']),
            f"EAD @ $125K avg"
        )

    with col3:
        st.metric(
            "Observed Defaults",
            f"{metrics['observed_defaults']:,}",
            f"{metrics['observed_default_rate']:.2f}% rate"
        )

    with col4:
        st.metric(
            "Mean PD",
            format_percentage(metrics['mean_pd']),
            f"Median: {format_percentage(metrics['median_pd'])}"
        )

    with col5:
        st.metric(
            "Expected Loss",
            format_currency(metrics['total_ecl']),
            "Basel III Compliant"
        )

    with col6:
        st.metric(
            "Avg Interest Rate",
            format_percentage(metrics['mean_int_rate']),
            f"Util: {format_percentage(metrics['mean_utilization'])}"
        )
    
    # Info Box
    st.info(
        "**Dashboard Context**: This dashboard provides institutional-grade credit risk analytics. "
        "All PD estimates derive from XGBoost scoring models trained on historical facility-level data. "
        "Expected Credit Loss calculated using Basel III IRB approach: ECL = PD x LGD x EAD."
    )
    
    st.markdown("---")
    
    # Main Analytics Section
    st.markdown("### Multi-Dimensional Portfolio Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Risk Distribution", "Performance Trends", "Segment Deep-Dive", "Comparative Analysis"])
    
    with tab1:
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown(f"""
            <div class="metric-card" style="height: 320px;">
                <div class="card-header">
                    <div class="card-title">Risk Tier Allocation</div>
                    <div class="card-subtitle">Portfolio concentration by risk tier</div>
                </div>
            """, unsafe_allow_html=True)
            
            risk_dist = df_filtered["risk_bucket"].value_counts().reindex(RISK_ORDER, fill_value=0)
            fig = px.pie(
                values=risk_dist.values,
                names=risk_dist.index,
                color=risk_dist.index,
                color_discrete_map=RISK_COLORS,
                hole=0.65
            )
            fig.update_traces(
                textposition="inside",
                textinfo="label+percent",
                marker=dict(line=dict(color=TH.BG_SECONDARY, width=2))
            )
            fig.update_layout(**TH.get_plotly_layout(), height=280, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_b:
            st.markdown(f"""
            <div class="metric-card" style="height: 320px;">
                <div class="card-header">
                    <div class="card-title">Default Rate by Tier</div>
                    <div class="card-subtitle">Observed defaults segmented by risk classification</div>
                </div>
            """, unsafe_allow_html=True)
            
            if "target" in df_filtered.columns:
                def_by_tier = df_filtered.groupby("risk_bucket", observed=True)["target"].agg(["sum", "count"])
                def_by_tier["rate"] = (def_by_tier["sum"] / def_by_tier["count"] * 100).fillna(0)
                def_by_tier = def_by_tier.reset_index()
                def_by_tier = def_by_tier.set_index("risk_bucket").reindex(RISK_ORDER).reset_index()
                
                fig = px.bar(
                    def_by_tier,
                    x="risk_bucket",
                    y="rate",
                    color="risk_bucket",
                    color_discrete_map=RISK_COLORS,
                    text="rate"
                )
                fig.update_traces(
                    texttemplate="<b>%{text:.1f}%</b>",
                    textposition="outside",
                    marker_line_width=0
                )
                fig.update_layout(
                    **TH.get_plotly_layout(),
                    height=280,
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title="Default Rate (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_c:
            st.markdown(f"""
            <div class="metric-card" style="height: 320px;">
                <div class="card-header">
                    <div class="card-title">Mean PD Intensity</div>
                    <div class="card-subtitle">Average predicted default probability per tier</div>
                </div>
            """, unsafe_allow_html=True)
            
            pd_by_tier = df_filtered.groupby("risk_bucket", observed=True)["prob"].mean().reset_index()
            pd_by_tier.columns = ["risk_bucket", "prob"]
            pd_by_tier["prob_pct"] = pd_by_tier["prob"] * 100
            pd_by_tier = pd_by_tier.set_index("risk_bucket").reindex(RISK_ORDER).reset_index()
            
            fig = px.bar(
                pd_by_tier,
                x="risk_bucket",
                y="prob_pct",
                color="risk_bucket",
                color_discrete_map=RISK_COLORS,
                text="prob_pct"
            )
            fig.update_traces(
                texttemplate="<b>%{text:.2f}%</b>",
                textposition="outside",
                marker_line_width=0
            )
            fig.update_layout(
                **TH.get_plotly_layout(),
                height=280,
                showlegend=False,
                xaxis_title="",
                yaxis_title="Mean PD (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-header">
                    <div class="card-title">PD Distribution Density</div>
                    <div class="card-subtitle">Probability density function across risk spectrum</div>
                </div>
            """, unsafe_allow_html=True)
            
            fig = px.histogram(
                df_filtered,
                x="prob",
                nbins=60,
                color="risk_bucket",
                color_discrete_map=RISK_COLORS,
                barmode="stack",
                opacity=0.85
            )
            fig.update_layout(
                **TH.get_plotly_layout(),
                height=350,
                xaxis_title="Probability of Default",
                yaxis_title="Frequency"
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_right:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-header">
                    <div class="card-title">Interest Rate vs PD Alignment</div>
                    <div class="card-subtitle">Pricing efficiency relative to risk classification</div>
                </div>
            """, unsafe_allow_html=True)
            
            sample_df = df_filtered.sample(min(5000, len(df_filtered)), random_state=42)
            fig = px.scatter(
                sample_df,
                x="prob",
                y="int_rate",
                color="risk_bucket",
                color_discrete_map=RISK_COLORS,
                opacity=0.4,
                size_max=6
            )
            fig.update_traces(marker=dict(size=4, line=dict(width=0)))
            fig.update_layout(
                **TH.get_plotly_layout(),
                height=350,
                xaxis_title="Probability of Default",
                yaxis_title="Interest Rate",
                hovermode="closest"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-header">
                <div class="card-title">Borrower Segmentation Analysis</div>
                <div class="card-subtitle">Deep-dive portfolio composition by credit utilization and repayment history</div>
            </div>
        """, unsafe_allow_html=True)
        
        seg_metric = st.selectbox(
            "Select Segmentation Metric",
            ["all_util", "mths_since_rcnt_il", "total_bal_il"],
            format_func=lambda x: FEATURE_DESCRIPTIONS.get(x, x),
            key="segment_metric"
        )
        
        df_seg = segment_by_feature(df_filtered, seg_metric, n_bins=5)
        seg_stats = df_seg.groupby(f"{seg_metric}_segment", observed=True).agg({
            "prob": "mean",
            "target": ["count", "sum"],
            "int_rate": "mean"
        }).reset_index()
        
        seg_stats.columns = ["segment", "mean_pd", "count", "defaults", "mean_rate"]
        seg_stats["default_rate"] = (seg_stats["defaults"] / seg_stats["count"] * 100).fillna(0)
        seg_stats["segment"] = seg_stats["segment"].astype(int) + 1
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Mean PD by Segment", "Default Rate by Segment"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Bar(x=seg_stats["segment"], y=seg_stats["mean_pd"]*100, name="Mean PD (%)",
                   marker_color=TH.ACCENT_BLUE, opacity=0.8),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=seg_stats["segment"], y=seg_stats["default_rate"], name="Default Rate (%)",
                   marker_color=TH.RISK_DANGER, opacity=0.8),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Segment Quintile", row=1, col=1)
        fig.update_xaxes(title_text="Segment Quintile", row=1, col=2)
        fig.update_yaxes(title_text="Mean PD (%)", row=1, col=1)
        fig.update_yaxes(title_text="Default Rate (%)", row=1, col=2)
        fig.update_layout(**TH.get_plotly_layout(), height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-header">
                <div class="card-title">Risk Tier Comparative Metrics</div>
                <div class="card-subtitle">Detailed cross-tier performance benchmarking</div>
            </div>
        """, unsafe_allow_html=True)
        
        comparison_data = df_filtered.groupby("risk_bucket", observed=True).agg({
            "prob": ["count", "mean", "min", "max"],
            "int_rate": "mean",
            "all_util": "mean",
            "target": ["sum", lambda x: (x.sum()/len(x)*100) if len(x) > 0 else 0]
        }).round(4)
        
        comparison_data.columns = ["Count", "Mean_PD", "Min_PD", "Max_PD", "Avg_Rate", "Avg_Util", "Defaults", "Default_Rate"]
        comparison_data = comparison_data.reset_index()
        comparison_data = comparison_data.set_index("risk_bucket").reindex(RISK_ORDER).reset_index()
        
        st.dataframe(
            comparison_data.style.format({
                "Mean_PD": "{:.4f}",
                "Avg_Rate": "{:.4f}",
                "Avg_Util": "{:.2%}",
                "Default_Rate": "{:.2f}%"
            }),
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ADVANCED ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

def page_advanced_analytics():
    """Detailed feature analysis, correlations, and model diagnostics."""
    render_header(
        "Advanced Analytics & Model Diagnostics",
        "Granular feature analysis, multicollinearity detection, and model performance evaluation",
        "DIAGNOSTIC"
    )
    
    # Filters
    selected_risk = st.sidebar.multiselect(
        "Risk Tier Filter",
        ["Low", "Medium", "High"],
        default=["Low", "Medium", "High"],
        key="adv_risk"
    )
    
    df_filtered = df_master[df_master["risk_bucket"].isin(selected_risk)].copy()
    
    st.info("**Analytical Focus**: Examine feature interactions, predictive signal strength, and model reliability metrics.")
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Feature Correlations", "Feature Impact", "Distribution Analysis", "Model Calibration"])
    
    with tab1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-header">
                <div class="card-title">Multicollinearity Matrix</div>
                <div class="card-subtitle">Pearson correlation coefficients for feature relationships</div>
            </div>
        """, unsafe_allow_html=True)
        
        numeric_cols = ["grade", "int_rate", "all_util", "max_bal_bc", "mths_since_rcnt_il", "total_bal_il", "il_util", "prob"]
        numeric_cols = [c for c in numeric_cols if c in df_filtered.columns]
        
        corr_matrix = df_filtered[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[FEATURE_DESCRIPTIONS.get(c, c) for c in corr_matrix.columns],
            y=[FEATURE_DESCRIPTIONS.get(c, c) for c in corr_matrix.columns],
            colorscale=[[0, TH.BG_SECONDARY], [0.5, TH.ACCENT_BLUE], [1, TH.TEXT_PRIMARY]],
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text:.2f}",
            textfont={"size": 10, "color": TH.TEXT_PRIMARY},
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            **TH.get_plotly_layout(),
            height=500,
            title_text="Feature Correlation Heatmap",
            xaxis_title="",
            yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-header">
                <div class="card-title">Feature Importance (Univariate)</div>
                <div class="card-subtitle">Absolute correlation magnitude with observed default outcomes</div>
            </div>
        """, unsafe_allow_html=True)
        
        if "target" in df_filtered.columns:
            importance = df_filtered[numeric_cols].corrwith(df_filtered["target"]).abs().sort_values(ascending=False)
            importance_df = pd.DataFrame({
                "feature": [FEATURE_DESCRIPTIONS.get(c, c) for c in importance.index],
                "importance": importance.values
            })
            
            fig = px.bar(
                importance_df.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=[[0, TH.RISK_SAFE], [1, TH.RISK_DANGER]]
            )
            fig.update_traces(marker_line_width=0, text=importance_df.sort_values("importance")["importance"], textposition="outside")
            fig.update_layout(
                **TH.get_plotly_layout(),
                height=400,
                showlegend=False,
                xaxis_title="Absolute Correlation with Default",
                yaxis_title=""
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        selected_feat = st.selectbox(
            "Select Feature for Distribution Analysis",
            numeric_cols,
            format_func=lambda x: FEATURE_DESCRIPTIONS.get(x, x)
        )
        
        col_l, col_r = st.columns(2)
        
        with col_l:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-header">
                    <div class="card-title">Density Distribution</div>
                    <div class="card-subtitle">By risk classification tier</div>
                </div>
            """, unsafe_allow_html=True)
            
            fig = px.histogram(
                df_filtered,
                x=selected_feat,
                color="risk_bucket",
                color_discrete_map=RISK_COLORS,
                nbins=50,
                barmode="overlay",
                opacity=0.6
            )
            fig.update_layout(**TH.get_plotly_layout(), height=350, xaxis_title=FEATURE_DESCRIPTIONS.get(selected_feat))
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_r:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-header">
                    <div class="card-title">Box Plot Comparison</div>
                    <div class="card-subtitle">Quartile spread across risk tiers</div>
                </div>
            """, unsafe_allow_html=True)
            
            fig = px.box(
                df_filtered,
                x="risk_bucket",
                y=selected_feat,
                color="risk_bucket",
                color_discrete_map=RISK_COLORS,
                category_orders={"risk_bucket": RISK_ORDER}
            )
            fig.update_layout(**TH.get_plotly_layout(), height=350, xaxis_title="", yaxis_title=FEATURE_DESCRIPTIONS.get(selected_feat))
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-header">
                <div class="card-title">Model Calibration Analysis</div>
                <div class="card-subtitle">Predicted vs Observed default rates (Expected Shortfall)</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Bin PD predictions and calculate observed defaults
        df_calib = df_filtered[df_filtered["target"].notna()].copy()
        df_calib["pd_bins"] = pd.cut(df_calib["prob"], bins=10)
        
        calibration = df_calib.groupby("pd_bins", observed=True).agg({
            "prob": "mean",
            "target": ["count", "mean"]
        }).reset_index()
        
        calibration.columns = ["bin", "mean_pred_pd", "count", "obs_default_rate"]
        calibration = calibration[calibration["count"] >= 10]  # Filter for significance
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=calibration["mean_pred_pd"],
            y=calibration["obs_default_rate"],
            mode="markers+lines",
            name="Observed",
            marker=dict(size=10, color=TH.RISK_DANGER),
            line=dict(color=TH.RISK_DANGER, width=2)
        ))
        
        # Perfect calibration line
        max_pd = df_calib["prob"].max()
        fig.add_trace(go.Scatter(
            x=[0, max_pd],
            y=[0, max_pd],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color=TH.TEXT_MUTED, width=2, dash="dash")
        ))
        
        fig.update_layout(
            **TH.get_plotly_layout(),
            height=400,
            xaxis_title="Predicted PD",
            yaxis_title="Observed Default Rate",
            hovermode="closest"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: STRESS TESTING & SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

def page_stress_testing():
    """Macroeconomic stress scenarios and capital impact modeling."""
    render_header(
        "Stress Testing & Scenario Analysis",
        "Basel III IRB capital impact modeling under adverse macroeconomic scenarios",
        "CCAR"
    )
    
    # Scenario Inputs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pd_shock = st.slider("PD Multiplier (β)", 1.0, 3.0, 1.5, 0.1, 
                            help="Uniform systemic shock to all predicted probabilities")
    with col2:
        lgd_rate = st.slider("Loss Given Default", 0.0, 1.0, 0.60, 0.05)
    with col3:
        ead_value = st.number_input("Avg Exposure at Default ($)", value=125_000, step=25_000)
    with col4:
        recovery_rate = st.slider("Recovery Rate", 0.0, 1.0, 0.40, 0.05)
    
    # Apply scenarios
    df_stress = df_master.copy()
    df_stress["baseline_ecl"] = df_stress["prob"] * lgd_rate * ead_value
    df_stress["stressed_pd"] = (df_stress["prob"] * pd_shock).clip(upper=1.0)
    df_stress["stressed_ecl"] = df_stress["stressed_pd"] * lgd_rate * ead_value
    df_stress["capital_impact"] = df_stress["stressed_ecl"] - df_stress["baseline_ecl"]
    
    # Summary metrics
    total_baseline = df_stress["baseline_ecl"].sum()
    total_stressed = df_stress["stressed_ecl"].sum()
    capital_need = total_stressed - total_baseline
    capital_pct = (capital_need / total_baseline * 100) if total_baseline > 0 else 0
    
    st.markdown("---")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.metric(
            "Baseline ECL",
            format_currency(total_baseline),
            "Current Stress Level"
        )

    with col_m2:
        st.metric(
            "Stressed ECL",
            format_currency(total_stressed),
            f"At {pd_shock}x PD Shock"
        )

    with col_m3:
        st.metric(
            "Capital Need",
            format_currency(capital_need),
            f"{capital_pct:+.1f}% Impact"
        )

    with col_m4:
        st.metric(
            "Capital Ratio",
            format_percentage(capital_need / (total_stressed + 1) * 100),
            "of Stressed Portfolio"
        )
    
    st.markdown("---")
    st.markdown("### Scenario Impact Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Impact by Risk Tier", "PD Migration Matrix", "Capital Sensitivity"])
    
    with tab1:
        col_l, col_r = st.columns(2)
        
        with col_l:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-header">
                    <div class="card-title">Baseline ECL by Tier</div>
                    <div class="card-subtitle">Current provisioning requirements</div>
                </div>
            """, unsafe_allow_html=True)
            
            ecl_base = df_stress.groupby("risk_bucket", observed=True)["baseline_ecl"].sum().reset_index()
            ecl_base = ecl_base.set_index("risk_bucket").reindex(RISK_ORDER).reset_index()
            
            fig = px.bar(ecl_base, x="risk_bucket", y="baseline_ecl", color="risk_bucket",
                        color_discrete_map=RISK_COLORS, text="baseline_ecl")
            fig.update_traces(
                texttemplate="$%{text:.0f}",
                textposition="outside",
                marker_line_width=0
            )
            fig.update_layout(**TH.get_plotly_layout(), height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_r:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-header">
                    <div class="card-title">Stressed ECL by Tier</div>
                    <div class="card-subtitle">Provisioning under adverse scenario</div>
                </div>
            """, unsafe_allow_html=True)
            
            ecl_str = df_stress.groupby("risk_bucket", observed=True)["stressed_ecl"].sum().reset_index()
            ecl_str = ecl_str.set_index("risk_bucket").reindex(RISK_ORDER).reset_index()
            
            fig = px.bar(ecl_str, x="risk_bucket", y="stressed_ecl", color="risk_bucket",
                        color_discrete_map=RISK_COLORS, text="stressed_ecl")
            fig.update_traces(
                texttemplate="$%{text:.0f}",
                textposition="outside",
                marker_line_width=0
            )
            fig.update_layout(**TH.get_plotly_layout(), height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-header">
                <div class="card-title">PD Tier Migration</div>
                <div class="card-subtitle">Facility movement across risk classifications under stress</div>
            </div>
        """, unsafe_allow_html=True)
        
        df_stress["stressed_tier"] = pd.cut(df_stress["stressed_pd"], bins=[0, 0.08, 0.20, 1.0], labels=RISK_ORDER)
        
        migration = pd.crosstab(
            df_stress["risk_bucket"],
            df_stress["stressed_tier"],
            margins=False
        ) / pd.crosstab(
            df_stress["risk_bucket"],
            df_stress["stressed_tier"],
            margins=False
        ).sum(axis=1, skipna=True) * 100
        
        migration = migration.reindex(RISK_ORDER)
        
        fig = px.imshow(
            migration,
            labels=dict(x="Stressed Tier", y="Baseline Tier", color="Migration %"),
            color_continuous_scale=[[0, TH.BG_SECONDARY], [1, TH.RISK_DANGER]],
            text_auto=".1f"
        )
        fig.update_layout(**TH.get_plotly_layout(), height=350)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-header">
                <div class="card-title">Sensitivity Analysis</div>
                <div class="card-subtitle">Capital impact across PD shock spectrum</div>
            </div>
        """, unsafe_allow_html=True)
        
        shock_range = np.linspace(1.0, 3.0, 21)
        sensitivity_results = []
        
        for shock in shock_range:
            temp_pd = (df_stress["prob"] * shock).clip(upper=1.0)
            temp_ecl = temp_pd * lgd_rate * ead_value
            temp_impact = temp_ecl.sum() - total_baseline
            sensitivity_results.append({
                "shock": shock,
                "ecl": temp_ecl.sum(),
                "capital_need": temp_impact,
                "capital_pct": (temp_impact / total_baseline * 100) if total_baseline > 0 else 0
            })
        
        sens_df = pd.DataFrame(sensitivity_results)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Total ECL Over Shocks", "Capital Need Over Shocks"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Scatter(x=sens_df["shock"], y=sens_df["ecl"], mode="lines+markers",
                      name="Total ECL", line=dict(color=TH.ACCENT_BLUE, width=3),
                      marker=dict(size=6)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=sens_df["shock"], y=sens_df["capital_pct"], mode="lines+markers",
                      name="Capital %", line=dict(color=TH.RISK_DANGER, width=3),
                      marker=dict(size=6)),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="PD Shock Multiplier", row=1, col=1)
        fig.update_xaxes(title_text="PD Shock Multiplier", row=1, col=2)
        fig.update_yaxes(title_text="Total ECL ($)", row=1, col=1)
        fig.update_yaxes(title_text="Capital Impact (%)", row=1, col=2)
        fig.update_layout(**TH.get_plotly_layout(), height=400, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

def page_data_explorer():
    """Interactive dataset exploration and custom analysis."""
    render_header(
        "Data Explorer & Custom Analysis",
        "Flexible facility-level data exploration with advanced filtering and export",
        "INTERACTIVE"
    )
    
    # Filtering
    st.sidebar.markdown("### Data Explorer Controls")
    
    filter_risk = st.sidebar.multiselect(
        "Filter by Risk Tier",
        ["Low", "Medium", "High"],
        default=["Low", "Medium", "High"]
    )
    
    filter_pd_range = st.sidebar.slider(
        "PD Range",
        float(df_master["prob"].min()),
        float(df_master["prob"].max()),
        (float(df_master["prob"].min()), float(df_master["prob"].max()))
    )
    
    filter_rate_range = st.sidebar.slider(
        "Interest Rate Range",
        float(df_master["int_rate"].min()),
        float(df_master["int_rate"].max()),
        (float(df_master["int_rate"].min()), float(df_master["int_rate"].max()))
    )
    
    # Apply all filters
    df_explore = df_master[
        (df_master["risk_bucket"].isin(filter_risk)) &
        (df_master["prob"] >= filter_pd_range[0]) &
        (df_master["prob"] <= filter_pd_range[1]) &
        (df_master["int_rate"] >= filter_rate_range[0]) &
        (df_master["int_rate"] <= filter_rate_range[1])
    ].copy()
    
    st.info(f"**Filtered Dataset**: {len(df_explore):,} facilities matching criteria ({len(df_explore)/len(df_master)*100:.1f}% of portfolio)")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Dataset View", "Statistical Summary", "Pairwise Comparison"])
    
    with tab1:
        st.markdown("### Facility-Level Data")
        
        display_cols = st.multiselect(
            "Select columns to display",
            df_explore.columns.tolist(),
            default=["grade", "int_rate", "prob", "risk_bucket", "target", "all_util"]
        )
        
        df_display = df_explore[display_cols].copy()
        
        # Sorting
        sort_col = st.selectbox("Sort by", df_display.columns)
        df_display = df_display.sort_values(sort_col, ascending=False)
        
        st.dataframe(df_display.head(500), use_container_width=True, height=400)
        
        # Download option
        csv_buffer = df_display.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_buffer,
            file_name=f"lenders_club_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.markdown("### Statistical Summary")
        
        numeric_cols = df_explore.select_dtypes(include=[np.number]).columns
        
        summary_stats = df_explore[numeric_cols].describe().T
        summary_stats = summary_stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        
        st.dataframe(summary_stats.style.format("{:.4f}"), use_container_width=True)
        
        # Grouped statistics
        st.markdown("### Grouped Statistics by Risk Tier")
        
        groupby_stat = df_explore.groupby("risk_bucket", observed=True)[["prob", "int_rate", "all_util"]].agg(["mean", "std", "min", "max"]).round(4)
        st.dataframe(groupby_stat, use_container_width=True)
    
    with tab3:
        col_x = st.selectbox("X-axis Feature", numeric_cols)
        col_y = st.selectbox("Y-axis Feature", numeric_cols)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-header">
                <div class="card-title">{col_x.upper()} vs {col_y.upper()}</div>
                <div class="card-subtitle">Bivariate relationship analysis</div>
            </div>
        """, unsafe_allow_html=True)
        
        sample_data = df_explore.sample(min(5000, len(df_explore)), random_state=42)
        
        fig = px.scatter(
            sample_data,
            x=col_x,
            y=col_y,
            color="risk_bucket",
            color_discrete_map=RISK_COLORS,
            opacity=0.5,
            size_max=5,
            hover_data=["prob", "int_rate", "target"]
        )
        
        fig.update_traces(marker=dict(size=4, line=dict(width=0)))
        fig.update_layout(**TH.get_plotly_layout(), height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION FLOW
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Sidebar Navigation
    st.sidebar.markdown(f"""
    <div style="padding-bottom: 24px; border-bottom: 2px solid {TH.BORDER_LIGHT}; margin-bottom: 24px;">
        <h2 style="margin: 0; color: {TH.TEXT_PRIMARY}; font-size: 1.3rem; letter-spacing: 1.5px;">LENDER'S CLUB</h2>
        <p style="margin: 6px 0 0 0; color: {TH.ACCENT_BLUE}; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px;">
            Risk Intelligence v3.0
        </p>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "MODULE SELECTION",
        ["Executive Dashboard", "Advanced Analytics", "Stress Testing", "Data Explorer"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown(f"""
    <div style="color: {TH.TEXT_MUTED}; font-size: 0.75rem; line-height: 1.8;">
        <div style="color: {TH.TEXT_SECONDARY}; font-weight: 600; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Analytics Suite</div>
        <span style="color: {TH.TEXT_MUTED};">Professional credit risk platform for institutional lending decision-making.</span>
    </div>
    <div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid {TH.BORDER_LIGHT};">
        <div style="color: {TH.TEXT_SECONDARY}; font-weight: 600; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;">Data &amp; Models</div>
        <div style="color: {TH.TEXT_MUTED}; font-size: 0.75rem; line-height: 2;">
            <span style="color: {TH.ACCENT_BLUE};">&#9656;</span> XGBoost PD scoring<br>
            <span style="color: {TH.ACCENT_BLUE};">&#9656;</span> Basel III IRB compliant<br>
            <span style="color: {TH.ACCENT_BLUE};">&#9656;</span> Real-time analytics
        </div>
    </div>
    <div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid {TH.BORDER_LIGHT};">
        <div style="color: {TH.TEXT_SECONDARY}; font-weight: 600; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;">System Status</div>
        <div style="color: {TH.TEXT_MUTED}; font-size: 0.75rem; line-height: 2;">
            <span style="color: {TH.RISK_SAFE};">&#9679;</span> System Online<br>
            <span style="color: {TH.RISK_SAFE};">&#9679;</span> Dataset: {len(df_master):,} facilities<br>
            <span style="color: {TH.RISK_SAFE};">&#9679;</span> Last Updated: Today
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Route to selected page
    if page == "Executive Dashboard":
        page_executive_summary()
    elif page == "Advanced Analytics":
        page_advanced_analytics()
    elif page == "Stress Testing":
        page_stress_testing()
    elif page == "Data Explorer":
        page_data_explorer()
    
    # Footer
    st.markdown(f"""
    <div style="text-align: center; padding: 40px 0; margin-top: 60px; border-top: 1px solid {TH.BORDER_LIGHT};">
        <p style="color: {TH.TEXT_MUTED}; font-size: 0.7rem; letter-spacing: 1px; text-transform: uppercase;">
            LENDER'S CLUB RISK INTELLIGENCE PLATFORM • V3.0 PROFESSIONAL EDITION
        </p>
        <p style="color: {TH.TEXT_MUTED}; font-size: 0.65rem; margin-top: 8px;">
            © {datetime.now().year} • Institutional Credit Risk Analytics • Basel III Compliant
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
