"""
═══════════════════════════════════════════════════════════════════════════════
LENDER'S CLUB - LOAN RISK ANALYSIS DASHBOARD v3.0
═══════════════════════════════════════════════════════════════════════════════

A credit risk analytics dashboard for loan portfolio management.
Uses ML-based default probability scoring to predict which borrowers
are likely to fail repayment, and estimates potential losses.

Version: 3.0
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
    page_title="Lender's Club - Loan Risk Dashboard",
    layout="wide",
    page_icon="LC",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Loan Risk Analysis Dashboard • v3.0"
    }
)

# ═══════════════════════════════════════════════════════════════════════════════
# PREMIUM THEME SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class ThemeConfig:
    """Dashboard color and design system."""
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

html, body, div, p, a, li, ul, ol, label, input, select, textarea, button,
h1, h2, h3, h4, h5, h6, th, td, caption, table {{
    font-family: 'Inter', -apple-system, sans-serif !important;
}}

html, body {{ background: {TH.BG_PRIMARY} !important; }}
.stApp {{ background: {TH.BG_PRIMARY} !important; }}

/* Hide Default Chrome */
#MainMenu, footer, .stAppDeployButton {{ visibility: hidden !important; }}
[data-testid="stHeaderActionElements"], [data-testid="manage-app-button"] {{ display: none !important; }}

/* Layout & Spacing */
.block-container {{ padding-top: 1rem !important; padding-bottom: 1rem !important; max-width: 1200px !important; }}
header[data-testid="stHeader"] {{ display: none !important; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {TH.BG_SECONDARY} !important;
    border-right: 1px solid {TH.BORDER_LIGHT} !important;
}}
section[data-testid="stSidebar"] .stRadio > div {{ gap: 2px !important; }}
section[data-testid="stSidebar"] .stRadio label {{
    padding: 8px 12px !important; border-radius: 6px !important;
    transition: background 0.2s !important;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    background: {TH.BG_HOVER} !important;
}}

/* Metric Cards */
div[data-testid="stMetric"] {{
    background: {TH.BG_SECONDARY};
    border: 1px solid {TH.BORDER_LIGHT};
    border-radius: 8px;
    padding: 20px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}}
div[data-testid="stMetric"]::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {TH.ACCENT_BLUE}, {TH.ACCENT_CYAN});
    opacity: 0;
    transition: opacity 0.3s;
}}
div[data-testid="stMetric"]:hover {{
    border-color: {TH.ACCENT_BLUE};
    background: {TH.BG_HOVER};
    box-shadow: 0 8px 32px rgba(59, 130, 246, 0.12);
    transform: translateY(-2px);
}}
div[data-testid="stMetric"]:hover::before {{ opacity: 1; }}
div[data-testid="stMetric"] label {{ color: {TH.TEXT_MUTED} !important; font-size: 0.7rem !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 1px; }}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {{ color: {TH.TEXT_PRIMARY} !important; font-weight: 700 !important; font-size: 1.6rem !important; font-family: 'JetBrains Mono', monospace !important; }}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {{ font-size: 0.7rem !important; }}

/* Typography */
h1, h2, h3, h4, h5, h6 {{ color: {TH.TEXT_PRIMARY} !important; }}
p, .stMarkdown {{ color: {TH.TEXT_PRIMARY} !important; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0px;
    border-bottom: 2px solid {TH.BORDER_LIGHT};
}}
.stTabs [data-baseweb="tab"] {{
    padding: 10px 20px;
    color: {TH.TEXT_MUTED} !important;
    font-weight: 600;
    font-size: 0.8rem;
    letter-spacing: 0.3px;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {TH.TEXT_PRIMARY} !important;
    background: {TH.BG_HOVER};
}}
.stTabs [aria-selected="true"] {{
    color: {TH.ACCENT_BLUE} !important;
    border-bottom: 2px solid {TH.ACCENT_BLUE} !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{
    background-color: {TH.ACCENT_BLUE} !important;
}}

/* Info box */
div[data-testid="stAlert"] {{
    background: rgba(59, 130, 246, 0.08) !important;
    border: 1px solid rgba(59, 130, 246, 0.2) !important;
    border-radius: 8px !important;
    color: {TH.TEXT_PRIMARY} !important;
}}

/* Controls */
.stSelectbox label, .stMultiSelect label, .stSlider label, .stNumberInput label {{
    color: {TH.TEXT_PRIMARY} !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px !important;
}}

/* Download button */
.stDownloadButton > button {{
    background: {TH.ACCENT_BLUE} !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
    transition: all 0.2s;
}}
.stDownloadButton > button:hover {{
    background: {TH.ACCENT_CYAN} !important;
    box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3) !important;
}}

/* Dataframes */
[data-testid="stDataFrame"] {{
    border: 1px solid {TH.BORDER_LIGHT} !important;
    border-radius: 8px !important;
}}

.status-badge {{
    display: inline-block;
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid {TH.ACCENT_BLUE};
    color: {TH.ACCENT_BLUE};
    font-size: 0.65rem;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace;
}}

/* Data source badge */
.data-source {{
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
}}
.data-source-title {{
    color: {TH.RISK_SAFE};
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}}
.data-source-text {{
    color: {TH.TEXT_SECONDARY};
    font-size: 0.8rem;
    line-height: 1.5;
}}

/* Scrollbars */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {TH.BORDER_LIGHT}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {TH.TEXT_MUTED}; }}

</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA MANAGEMENT LAYER
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading dataset...")
def load_master_dataset() -> pd.DataFrame:
    """Load and validate master lending dataset."""
    fp = os.path.join(os.path.dirname(__file__), "final_dataset.csv")
    if not os.path.exists(fp):
        st.error("CRITICAL: Master dataset not found. Verify upstream pipeline.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(fp)
        logger.info(f"✓ Dataset loaded: {len(df):,} loans")
        return df
    except Exception as e:
        logger.error(f"✗ Ingestion error: {str(e)}")
        return pd.DataFrame()

# Feature Metadata
FEATURE_DESCRIPTIONS = {
    "grade": "Credit Grade",
    "int_rate": "Interest Rate (%)",
    "all_util": "Credit Utilization (%)",
    "max_bal_bc": "Max Credit Card Balance",
    "mths_since_rcnt_il": "Months Since Last Installment",
    "total_bal_il": "Total Installment Balance",
    "il_util": "Installment Utilization (%)",
    "prob": "Default Probability",
    "target": "Actually Defaulted (1=Yes)",
    "risk_bucket": "Risk Level",
    "ECL": "Expected Loss",
    "stressed_ECL": "Stressed Loss"
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

# Feature columns used for model training
FEATURE_COLS = ["grade", "int_rate", "all_util", "max_bal_bc",
                "mths_since_rcnt_il", "total_bal_il", "il_util"]

SHAP_SAMPLE_SIZE = 3000  # Compute SHAP on a sample for speed

@st.cache_resource(show_spinner="Training model & computing SHAP (one-time, ~10s)...")
def train_xgb_model():
    """Train XGBoost with train/test split, compute metrics and SHAP."""
    df = load_master_dataset()
    X = df[FEATURE_COLS].copy()
    y = df["target"]

    # Proper train/test split for honest evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pos = y_train.sum()
    neg = len(y_train) - pos
    scale = neg / pos if pos > 0 else 1

    model = XGBClassifier(
        objective="binary:logistic", eval_metric="auc",
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale, random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate on TEST set (unseen data — honest metrics)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "auc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": cm,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    # SHAP on a sample for performance
    sample_idx = np.random.RandomState(42).choice(len(X), SHAP_SAMPLE_SIZE, replace=False)
    X_sample = X.iloc[sample_idx]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    return model, explainer, shap_values, X_sample, sample_idx, metrics

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

def render_section(title: str, subtitle: str = "") -> None:
    """Render a chart section title with subtitle."""
    st.markdown(f"""
    <div style="margin-bottom: 8px;">
        <div style="color: {TH.TEXT_PRIMARY}; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;">{title}</div>
        <div style="color: {TH.TEXT_SECONDARY}; font-size: 0.75rem; margin-top: 2px;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def render_header(title: str, subtitle: str = "", status: str = "ACTIVE") -> None:
    """Render page header with styling."""
    st.markdown(f"""
    <div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid {TH.BORDER_LIGHT};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin: 0; color: {TH.TEXT_PRIMARY}; font-size: 1.3rem;">{title}</h3>
                {f'<p style="color: {TH.TEXT_SECONDARY}; margin: 4px 0 0 0; font-size: 0.8rem;">{subtitle}</p>' if subtitle else ''}
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
        "Loan Portfolio Overview",
        "Key metrics on loan health, default rates, and predicted losses",
        "LIVE"
    )

    # Model & data credibility strip
    _, _, _, _, _, model_metrics = train_xgb_model()

    st.markdown(f"""
    <div style="display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap;">
        <div class="data-source" style="flex: 1; min-width: 280px; margin-bottom: 0;">
            <div class="data-source-title">Data Source</div>
            <div class="data-source-text">
                <strong>Lending Club via Kaggle</strong> — real-world peer-to-peer lending data<br>
                <strong>30,000</strong> loans · <strong>7 features</strong> · Actual defaults (not synthetic)
            </div>
        </div>
        <div style="flex: 1; min-width: 280px; background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.2); border-radius: 8px; padding: 16px 20px;">
            <div style="color: {TH.ACCENT_BLUE}; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">Model Performance (Test Set)</div>
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div>
                    <div style="color: {TH.TEXT_PRIMARY}; font-size: 1.3rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{model_metrics['auc']:.3f}</div>
                    <div style="color: {TH.TEXT_MUTED}; font-size: 0.65rem; font-weight: 600; text-transform: uppercase;">ROC-AUC</div>
                </div>
                <div>
                    <div style="color: {TH.TEXT_PRIMARY}; font-size: 1.3rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{model_metrics['accuracy']*100:.1f}%</div>
                    <div style="color: {TH.TEXT_MUTED}; font-size: 0.65rem; font-weight: 600; text-transform: uppercase;">Accuracy</div>
                </div>
                <div>
                    <div style="color: {TH.TEXT_PRIMARY}; font-size: 1.3rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{model_metrics['f1']:.3f}</div>
                    <div style="color: {TH.TEXT_MUTED}; font-size: 0.65rem; font-weight: 600; text-transform: uppercase;">F1 Score</div>
                </div>
                <div>
                    <div style="color: {TH.TEXT_MUTED}; font-size: 0.7rem; margin-top: 4px;">XGBoost · 80/20 split<br>{model_metrics['train_size']:,} train · {model_metrics['test_size']:,} test</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    selected_risk = st.sidebar.multiselect(
        "Risk Level Filter",
        ["Low", "Medium", "High"],
        default=["Low", "Medium", "High"],
        key="exec_risk"
    )

    pd_range = st.sidebar.slider(
        "Default Probability Range",
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

    # KPI Cards - Row 1
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Loans",
            f"{metrics['total_facilities']:,}",
            f"{(len(df_filtered)/len(df_master)*100):.1f}% of portfolio"
        )

    with col2:
        st.metric(
            "Exposure",
            format_currency(metrics['total_exposure']),
            "Avg $125K/loan"
        )

    with col3:
        st.metric(
            "Defaults",
            f"{metrics['observed_defaults']:,}",
            f"{metrics['observed_default_rate']:.2f}% rate"
        )

    # KPI Cards - Row 2
    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            "Avg PD",
            format_percentage(metrics['mean_pd']),
            f"Median: {format_percentage(metrics['median_pd'])}"
        )

    with col5:
        st.metric(
            "Expected Loss",
            format_currency(metrics['total_ecl']),
            "Prob × Loss Rate × Amt"
        )

    with col6:
        st.metric(
            "Avg Rate",
            format_percentage(metrics['mean_int_rate']),
            f"Util: {format_percentage(metrics['mean_utilization'])}"
        )

    # Info Box
    st.info(
        "**How to read this**: Default Probability is predicted by an XGBoost ML model. "
        "Expected Loss = Default Probability × Loss Rate (60%) × Loan Amount ($125K). "
        "Higher probability means the borrower is more likely to fail repayment."
    )
    
    # Main Analytics Section
    st.markdown("### Portfolio Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Risk Distribution", "Performance Trends", "Segment Deep-Dive", "Comparative Analysis"])
    
    with tab1:
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            render_section("Risk Tier Allocation", "Portfolio concentration by risk tier")

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
        
        with col_b:
            render_section("Default Rate by Tier", "Observed defaults segmented by risk classification")

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
        
        with col_c:
            render_section("Mean Default Probability", "Average predicted default chance per risk tier")

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
                yaxis_title="Avg Default Prob (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col_left, col_right = st.columns(2)
        
        with col_left:
            render_section("Default Probability Distribution", "Spread of predicted default probabilities across all loans")

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
        
        with col_right:
            render_section("Interest Rate vs Default Probability", "Relationship between loan pricing and predicted risk")

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
    
    with tab3:
        render_section("Borrower Segments", "Grouping borrowers by key features and comparing default patterns")

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
            subplot_titles=("Avg Default Prob by Segment", "Default Rate by Segment"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Bar(x=seg_stats["segment"], y=seg_stats["mean_pd"]*100, name="Avg Default Prob (%)",
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
        fig.update_yaxes(title_text="Avg Default Prob (%)", row=1, col=1)
        fig.update_yaxes(title_text="Default Rate (%)", row=1, col=2)
        fig.update_layout(**TH.get_plotly_layout(), height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        render_section("Risk Level Comparison", "Side-by-side stats for each risk group")

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


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ADVANCED ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

def page_advanced_analytics():
    """Detailed feature analysis, correlations, and model diagnostics."""
    render_header(
        "Advanced Analytics & Model Diagnostics",
        "Feature analysis, correlation detection, and model performance evaluation",
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
    
    st.info("**Focus**: Examine how features relate to each other, which features best predict defaults, and how reliable the model is.")
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Feature Correlations", "Feature Impact", "Distribution Analysis", "Model Calibration"])
    
    with tab1:
        render_section("Correlation Matrix", "How strongly each feature is related to every other feature")

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

    with tab2:
        render_section("Feature Importance", "Which features have the strongest relationship with actual defaults")

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

    with tab3:
        selected_feat = st.selectbox(
            "Select Feature for Distribution Analysis",
            numeric_cols,
            format_func=lambda x: FEATURE_DESCRIPTIONS.get(x, x)
        )
        
        col_l, col_r = st.columns(2)
        
        with col_l:
            render_section("Density Distribution", "By risk classification tier")

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

        with col_r:
            render_section("Box Plot Comparison", "Quartile spread across risk tiers")

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

    with tab4:
        render_section("Model Calibration", "How well the model's predictions match actual default rates")

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
            xaxis_title="Predicted Default Probability",
            yaxis_title="Actual Default Rate",
            hovermode="closest"
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: STRESS TESTING & SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

def page_stress_testing():
    """Simulate worst-case scenarios and measure loss impact."""
    render_header(
        "Stress Testing & What-If Scenarios",
        "Simulate worst-case conditions and see how they affect expected losses",
        "SCENARIO"
    )
    
    # Scenario Inputs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pd_shock = st.slider("Risk Multiplier", 1.0, 3.0, 1.5, 0.1,
                            help="How much to increase all default probabilities (e.g. 1.5x = 50% worse)")
    with col2:
        lgd_rate = st.slider("Loss Rate per Default", 0.0, 1.0, 0.60, 0.05,
                            help="What % of the loan amount is lost when a borrower defaults")
    with col3:
        ead_value = st.number_input("Avg Loan Amount ($)", value=125_000, step=25_000,
                                   help="Average loan size used to calculate total losses")
    with col4:
        recovery_rate = st.slider("Recovery Rate", 0.0, 1.0, 0.40, 0.05,
                                 help="What % of the defaulted amount can be recovered")
    
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
            "Normal Loss",
            format_currency(total_baseline),
            "Under current conditions"
        )

    with col_m2:
        st.metric(
            "Stressed Loss",
            format_currency(total_stressed),
            f"At {pd_shock}x risk increase"
        )

    with col_m3:
        st.metric(
            "Additional Loss",
            format_currency(capital_need),
            f"{capital_pct:+.1f}% increase"
        )

    with col_m4:
        st.metric(
            "Loss Ratio",
            format_percentage(capital_need / (total_stressed + 1) * 100),
            "Additional vs Total"
        )
    
    st.markdown("---")
    st.markdown("### Scenario Impact Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Loss by Risk Tier", "Risk Tier Shifts", "Sensitivity Analysis"])
    
    with tab1:
        col_l, col_r = st.columns(2)
        
        with col_l:
            render_section("Normal Loss by Tier", "Expected losses under current conditions")

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

        with col_r:
            render_section("Stressed Loss by Tier", "Expected losses under worst-case scenario")

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

    with tab2:
        render_section("Risk Tier Shifts", "How loans move between risk levels under stress")

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
            labels=dict(x="Stressed Tier", y="Original Tier", color="% of Loans"),
            color_continuous_scale=[[0, TH.BG_SECONDARY], [1, TH.RISK_DANGER]],
            text_auto=".1f"
        )
        fig.update_layout(**TH.get_plotly_layout(), height=350)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        render_section("Sensitivity Analysis", "How losses change as risk multiplier increases")

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
            subplot_titles=("Total Expected Loss vs Risk Multiplier", "Additional Loss vs Risk Multiplier"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Scatter(x=sens_df["shock"], y=sens_df["ecl"], mode="lines+markers",
                      name="Total Loss", line=dict(color=TH.ACCENT_BLUE, width=3),
                      marker=dict(size=6)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=sens_df["shock"], y=sens_df["capital_pct"], mode="lines+markers",
                      name="Capital %", line=dict(color=TH.RISK_DANGER, width=3),
                      marker=dict(size=6)),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Risk Multiplier", row=1, col=1)
        fig.update_xaxes(title_text="Risk Multiplier", row=1, col=2)
        fig.update_yaxes(title_text="Total Expected Loss ($)", row=1, col=1)
        fig.update_yaxes(title_text="Additional Loss (%)", row=1, col=2)
        fig.update_layout(**TH.get_plotly_layout(), height=400, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

def page_data_explorer():
    """Interactive dataset exploration and custom analysis."""
    render_header(
        "Data Explorer",
        "Browse, filter, and export loan-level data",
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
        "Default Probability Range",
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
    
    st.info(f"**Filtered Dataset**: {len(df_explore):,} loans matching criteria ({len(df_explore)/len(df_master)*100:.1f}% of portfolio)")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Dataset View", "Statistical Summary", "Pairwise Comparison"])
    
    with tab1:
        st.markdown("### Loan-Level Data")
        
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
        
        render_section(f"{col_x.upper()} vs {col_y.upper()}", "How these two features relate to each other")

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


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL EXPLAINABILITY (SHAP)
# ═══════════════════════════════════════════════════════════════════════════════

def page_explainability():
    """SHAP-based model explainability — WHY a loan is risky."""
    render_header(
        "Model Explainability (SHAP)",
        "Understand which features drive the model's predictions and why each loan is classified as risky or safe",
        "XAI"
    )

    model, explainer, shap_values, X_shap, sample_idx, model_metrics = train_xgb_model()
    feature_labels = [FEATURE_DESCRIPTIONS.get(c, c) for c in FEATURE_COLS]

    # Model card
    cm = model_metrics["confusion_matrix"]
    st.markdown(f"""
    <div style="background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.2); border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
            <div>
                <div style="color: {TH.ACCENT_BLUE}; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Model Card</div>
                <div style="color: {TH.TEXT_PRIMARY}; font-size: 0.95rem; font-weight: 700; margin-top: 4px;">XGBoost Classifier</div>
                <div style="color: {TH.TEXT_MUTED}; font-size: 0.72rem; margin-top: 2px;">200 trees · max depth 3 · learning rate 0.05 · 80/20 train-test split</div>
            </div>
            <div style="display: flex; gap: 24px; flex-wrap: wrap;">
                <div style="text-align: center;">
                    <div style="color: {TH.RISK_SAFE}; font-size: 1.4rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{model_metrics['auc']:.3f}</div>
                    <div style="color: {TH.TEXT_MUTED}; font-size: 0.6rem; font-weight: 700; text-transform: uppercase;">AUC</div>
                </div>
                <div style="text-align: center;">
                    <div style="color: {TH.TEXT_PRIMARY}; font-size: 1.4rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{model_metrics['precision']:.3f}</div>
                    <div style="color: {TH.TEXT_MUTED}; font-size: 0.6rem; font-weight: 700; text-transform: uppercase;">Precision</div>
                </div>
                <div style="text-align: center;">
                    <div style="color: {TH.TEXT_PRIMARY}; font-size: 1.4rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{model_metrics['recall']:.3f}</div>
                    <div style="color: {TH.TEXT_MUTED}; font-size: 0.6rem; font-weight: 700; text-transform: uppercase;">Recall</div>
                </div>
                <div style="text-align: center;">
                    <div style="color: {TH.TEXT_PRIMARY}; font-size: 1.4rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{model_metrics['f1']:.3f}</div>
                    <div style="color: {TH.TEXT_MUTED}; font-size: 0.6rem; font-weight: 700; text-transform: uppercase;">F1</div>
                </div>
            </div>
        </div>
        <div style="color: {TH.TEXT_MUTED}; font-size: 0.7rem; margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(59,130,246,0.15);">
            Confusion Matrix (test set) — TN: {cm[0][0]:,} · FP: {cm[0][1]:,} · FN: {cm[1][0]:,} · TP: {cm[1][1]:,}
            &nbsp;|&nbsp; Data: Lending Club via Kaggle · {model_metrics['train_size']:,} train · {model_metrics['test_size']:,} test
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "**What is SHAP?** SHAP (SHapley Additive exPlanations) shows how much each feature "
        "pushes a prediction higher or lower. Red = increases default risk, Blue = decreases it."
    )

    tab1, tab2, tab3 = st.tabs(["Global Feature Impact", "Individual Loan Explanation", "Feature Dependence"])

    # ── Tab 1: Global SHAP summary ──
    with tab1:
        render_section("Global Feature Importance", "Which features matter most across ALL loans")

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        imp_df = pd.DataFrame({
            "Feature": feature_labels,
            "Mean |SHAP|": mean_abs_shap
        }).sort_values("Mean |SHAP|", ascending=True)

        fig = px.bar(
            imp_df, x="Mean |SHAP|", y="Feature", orientation="h",
            color="Mean |SHAP|",
            color_continuous_scale=[[0, TH.ACCENT_CYAN], [1, TH.RISK_DANGER]]
        )
        fig.update_traces(marker_line_width=0)
        fig.update_layout(
            **TH.get_plotly_layout(), height=400, showlegend=False,
            xaxis_title="Average Impact on Prediction",
            yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        render_section("SHAP Bee Swarm", "Each dot is one loan — position shows impact, color shows feature value")

        # Build bee-swarm using px.scatter (compatible with all Plotly versions)
        n_swarm = min(1500, len(X_shap))
        swarm_idx = np.random.RandomState(42).choice(len(X_shap), n_swarm, replace=False)

        swarm_rows = []
        for i, col in enumerate(FEATURE_COLS):
            for j in swarm_idx:
                swarm_rows.append({
                    "Feature": feature_labels[i],
                    "SHAP Value": float(shap_values[j, i]),
                    "Feature Value": float(X_shap.iloc[j][col])
                })
        swarm_df = pd.DataFrame(swarm_rows)

        fig = px.scatter(
            swarm_df, x="SHAP Value", y="Feature", color="Feature Value",
            color_continuous_scale=[[0, TH.ACCENT_BLUE], [1, TH.RISK_DANGER]]
        )
        fig.update_traces(marker=dict(size=3, opacity=0.4))
        fig.update_layout(
            **TH.get_plotly_layout(), height=450,
            xaxis_title="SHAP Value (impact on prediction)",
            yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Individual loan explanation ──
    with tab2:
        render_section("Explain a Single Loan", "See exactly WHY the model scored this loan the way it did")

        col1, col2 = st.columns([1, 2])
        with col1:
            loan_idx = st.number_input(
                "Loan Index", min_value=0, max_value=len(df_master)-1,
                value=0, step=1, help="Pick any loan from the dataset"
            )

        loan_row = df_master.iloc[loan_idx]
        # Compute SHAP for this specific loan on-the-fly (fast for single row)
        loan_X = df_master[FEATURE_COLS].iloc[[loan_idx]]
        loan_shap = explainer.shap_values(loan_X)[0]
        base_val = explainer.expected_value
        if isinstance(base_val, np.ndarray):
            base_val = float(base_val[0])

        pred_prob = float(model.predict_proba(loan_X)[0][1])
        risk = loan_row.get("risk_bucket", "Unknown")
        risk_color = RISK_COLORS.get(risk, TH.TEXT_PRIMARY)

        with col2:
            st.markdown(f"""
            <div style="display:flex; gap:32px; align-items:center; padding:12px 0;">
                <div><div style="color:{TH.TEXT_MUTED}; font-size:0.75rem;">PREDICTED DEFAULT PROB</div>
                <div style="font-size:1.5rem; font-weight:700; color:{TH.TEXT_PRIMARY};">{pred_prob*100:.1f}%</div></div>
                <div><div style="color:{TH.TEXT_MUTED}; font-size:0.75rem;">RISK LEVEL</div>
                <div style="font-size:1.5rem; font-weight:700; color:{risk_color};">{risk}</div></div>
                <div><div style="color:{TH.TEXT_MUTED}; font-size:0.75rem;">ACTUAL DEFAULT</div>
                <div style="font-size:1.5rem; font-weight:700; color:{TH.TEXT_PRIMARY};">{'Yes' if loan_row.get('target', 0) == 1 else 'No'}</div></div>
            </div>
            """, unsafe_allow_html=True)

        # Waterfall chart
        contributions = sorted(
            zip(feature_labels, loan_shap, [float(loan_row[c]) for c in FEATURE_COLS]),
            key=lambda x: abs(x[1]), reverse=True
        )

        feat_names = [c[0] for c in contributions]
        feat_shap_vals = [c[1] for c in contributions]
        feat_vals = [c[2] for c in contributions]
        colors = [TH.RISK_DANGER if s > 0 else TH.ACCENT_BLUE for s in feat_shap_vals]

        fig = go.Figure(go.Bar(
            x=feat_shap_vals, y=[f"{n} = {v:.3f}" for n, v in zip(feat_names, feat_vals)],
            orientation="h", marker_color=colors
        ))
        layout = TH.get_plotly_layout()
        fig.update_layout(
            **layout, height=380,
            xaxis_title="SHAP Value (red = increases risk, blue = decreases)",
            yaxis_title=""
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Base prediction: {base_val*100:.1f}% — features push it to {pred_prob*100:.1f}%. "
            "Red bars increase default risk. Blue bars decrease it."
        )

    # ── Tab 3: Feature dependence ──
    with tab3:
        render_section("Feature Dependence", "How one feature's value affects predictions — colored by a second feature")

        col1, col2 = st.columns(2)
        with col1:
            dep_feat = st.selectbox(
                "Primary Feature", FEATURE_COLS,
                format_func=lambda x: FEATURE_DESCRIPTIONS.get(x, x),
                key="shap_dep_feat"
            )
        with col2:
            interaction_feat = st.selectbox(
                "Color By Feature", [c for c in FEATURE_COLS if c != dep_feat],
                format_func=lambda x: FEATURE_DESCRIPTIONS.get(x, x),
                key="shap_int_feat"
            )

        feat_idx = FEATURE_COLS.index(dep_feat)

        fig = px.scatter(
            x=X_shap[dep_feat].values,
            y=shap_values[:, feat_idx],
            color=X_shap[interaction_feat].values,
            color_continuous_scale=[[0, TH.ACCENT_BLUE], [1, TH.RISK_DANGER]],
            opacity=0.5,
            labels={
                "x": FEATURE_DESCRIPTIONS.get(dep_feat, dep_feat),
                "y": f"SHAP Value for {FEATURE_DESCRIPTIONS.get(dep_feat, dep_feat)}",
                "color": FEATURE_DESCRIPTIONS.get(interaction_feat, interaction_feat)
            }
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(**TH.get_plotly_layout(), height=420)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION FLOW
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Sidebar Navigation
    st.sidebar.markdown(f"""
    <div style="padding-bottom: 24px; border-bottom: 2px solid {TH.BORDER_LIGHT}; margin-bottom: 24px;">
        <h2 style="margin: 0; color: {TH.TEXT_PRIMARY}; font-size: 1.3rem; letter-spacing: 1.5px;">LENDER'S CLUB</h2>
        <p style="margin: 6px 0 0 0; color: {TH.ACCENT_BLUE}; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px;">
            Loan Risk Dashboard v3.0
        </p>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "MODULE SELECTION",
        ["Executive Dashboard", "Advanced Analytics", "Model Explainability", "Stress Testing", "Data Explorer"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown(f"""
    <div style="background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2); border-radius: 6px; padding: 12px; margin-bottom: 12px;">
        <div style="color: {TH.RISK_SAFE}; font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Data Source</div>
        <div style="color: {TH.TEXT_PRIMARY}; font-size: 0.8rem; font-weight: 600; margin-top: 4px;">Lending Club via Kaggle (Real Data)</div>
        <div style="color: {TH.TEXT_MUTED}; font-size: 0.7rem; margin-top: 2px;">{len(df_master):,} loans · 7 features · Actual defaults</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <div style="background: {TH.BG_HOVER}; border-radius: 6px; padding: 12px; margin-bottom: 12px;">
        <div style="color: {TH.TEXT_MUTED}; font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Technology Stack</div>
        <div style="color: {TH.TEXT_SECONDARY}; font-size: 0.72rem; margin-top: 6px; line-height: 1.7;">
            XGBoost Classifier · SHAP Explainability<br>
            Plotly Visualizations · Streamlit Framework
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <div style="color: {TH.TEXT_MUTED}; font-size: 0.7rem; line-height: 1.8; padding: 4px 0;">
        <span style="color: {TH.RISK_SAFE};">●</span> System Online &nbsp;
        <span style="color: {TH.RISK_SAFE};">●</span> Dataset Loaded
    </div>
    """, unsafe_allow_html=True)
    
    # Route to selected page
    if page == "Executive Dashboard":
        page_executive_summary()
    elif page == "Advanced Analytics":
        page_advanced_analytics()
    elif page == "Model Explainability":
        page_explainability()
    elif page == "Stress Testing":
        page_stress_testing()
    elif page == "Data Explorer":
        page_data_explorer()
    
    # Footer
    st.markdown(f"""
    <div style="text-align: center; padding: 40px 0; margin-top: 60px; border-top: 1px solid {TH.BORDER_LIGHT};">
        <p style="color: {TH.TEXT_MUTED}; font-size: 0.7rem; letter-spacing: 1px; text-transform: uppercase;">
            LENDER'S CLUB • LOAN RISK ANALYSIS DASHBOARD • V3.0
        </p>
        <p style="color: {TH.TEXT_MUTED}; font-size: 0.65rem; margin-top: 8px;">
            Built by Aditya Parbhakar • Data: Lending Club (Kaggle) • XGBoost + SHAP
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
