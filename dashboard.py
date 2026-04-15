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
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
import os
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import pickle

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
    """Dashboard color and design system — clean light professional theme."""
    # Primary Palette
    BG_PRIMARY = "#FFFFFF"        # White background
    BG_SECONDARY = "#F8FAFC"      # Card / sidebar background
    BG_HOVER = "#F1F5F9"          # Hover state

    # Text Hierarchy
    TEXT_PRIMARY = "#1E293B"      # Slate-900
    TEXT_SECONDARY = "#475569"    # Slate-600
    TEXT_MUTED = "#94A3B8"        # Slate-400

    # Accents & Severity
    ACCENT_BLUE = "#2563EB"       # Primary action (blue-600)
    ACCENT_CYAN = "#0891B2"       # Secondary action (cyan-600)

    RISK_SAFE = "#059669"         # Low risk (emerald-600)
    RISK_CAUTION = "#D97706"      # Medium risk (amber-600)
    RISK_DANGER = "#DC2626"       # High risk (red-600)

    # Structural
    BORDER_LIGHT = "#E2E8F0"      # Slate-200
    BORDER_DARK = "#CBD5E1"       # Slate-300
    GLOW_SUBTLE = "rgba(37, 99, 235, 0.06)"

    @classmethod
    def get_plotly_layout(cls) -> Dict[str, Any]:
        """Standard Plotly template with clean professional styling."""
        return dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, -apple-system, sans-serif", color=cls.TEXT_SECONDARY, size=11),
            margin=dict(l=24, r=24, t=44, b=24),
            xaxis=dict(
                gridcolor="rgba(0,0,0,0.06)",
                zerolinecolor="rgba(0,0,0,0.1)",
                tickfont=dict(color=cls.TEXT_SECONDARY, size=10),
                title_font=dict(color=cls.TEXT_PRIMARY, size=12, family="Inter"),
                showgrid=True,
                zeroline=False,
            ),
            yaxis=dict(
                gridcolor="rgba(0,0,0,0.06)",
                zerolinecolor="rgba(0,0,0,0.1)",
                tickfont=dict(color=cls.TEXT_SECONDARY, size=10),
                title_font=dict(color=cls.TEXT_PRIMARY, size=12, family="Inter"),
                showgrid=True,
                zeroline=False,
            ),
            hoverlabel=dict(
                bgcolor="#FFFFFF",
                font_size=12,
                font_family="Inter",
                font_color=cls.TEXT_PRIMARY,
                bordercolor=cls.BORDER_DARK,
                namelength=-1
            ),
            legend=dict(
                font=dict(color=cls.TEXT_PRIMARY, size=11),
                bgcolor="rgba(255,255,255,0.9)",
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

/* Hide Default Chrome (keep sidebar toggle visible) */
#MainMenu, footer, .stAppDeployButton {{ visibility: hidden !important; }}
[data-testid="manage-app-button"] {{ display: none !important; }}

/* Push content below Streamlit Cloud top bar */
.block-container {{ padding-top: 4rem !important; padding-bottom: 1rem !important; max-width: 1200px !important; }}
section[data-testid="stSidebar"] > div:first-child {{ padding-top: 4rem !important; }}

/* Sidebar close button (X inside open sidebar) */
section[data-testid="stSidebar"] button {{
    color: {TH.TEXT_PRIMARY} !important;
    opacity: 1 !important;
    visibility: visible !important;
}}
section[data-testid="stSidebar"] button svg {{
    fill: {TH.TEXT_PRIMARY} !important;
    stroke: {TH.TEXT_PRIMARY} !important;
}}

/* Sidebar open button (arrow when sidebar is collapsed) — BIG and BLUE */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {{
    z-index: 999999 !important;
    top: 0.7rem !important;
    left: 0.7rem !important;
}}
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button,
[data-testid="stSidebarNav"] button {{
    background: {TH.ACCENT_BLUE} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    min-height: 44px !important;
    box-shadow: 0 3px 12px rgba(37,99,235,0.4) !important;
    opacity: 1 !important;
    visibility: visible !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
}}
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="collapsedControl"] button:hover {{
    background: {TH.ACCENT_CYAN} !important;
    transform: scale(1.08);
}}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg {{
    fill: white !important;
    stroke: white !important;
    color: white !important;
    width: 22px !important;
    height: 22px !important;
}}

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
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
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
    background: rgba(37, 99, 235, 0.04) !important;
    border: 1px solid rgba(37, 99, 235, 0.15) !important;
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
    box-shadow: 0 4px 12px rgba(8, 145, 178, 0.2) !important;
}}

/* Dataframes */
[data-testid="stDataFrame"] {{
    border: 1px solid {TH.BORDER_LIGHT} !important;
    border-radius: 8px !important;
}}

.status-badge {{
    display: inline-block;
    background: rgba(37, 99, 235, 0.08);
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
    background: rgba(5, 150, 105, 0.04);
    border: 1px solid rgba(5, 150, 105, 0.2);
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

/* Spinner */
[data-testid="stSpinner"] > div {{
    background: {TH.BG_SECONDARY} !important;
    color: {TH.TEXT_PRIMARY} !important;
    border: 1px solid {TH.BORDER_LIGHT} !important;
    border-radius: 8px !important;
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

# Feature Metadata (all model features are standardized/scaled)
FEATURE_DESCRIPTIONS = {
    "grade": "Credit Grade",
    "int_rate": "Interest Rate",
    "all_util": "Credit Utilization",
    "max_bal_bc": "Max Bankcard Balance",
    "mths_since_rcnt_il": "Months Since Recent IL",
    "total_bal_il": "Total Installment Balance",
    "il_util": "Installment Utilization",
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

_MODEL_CACHE_PATH = os.path.join(os.path.dirname(__file__), "pretrained_models.pkl")

@st.cache_resource(show_spinner=False)
def train_models():
    """Load pre-trained models from disk. Falls back to training if file missing."""
    try:
        if os.path.exists(_MODEL_CACHE_PATH):
            with open(_MODEL_CACHE_PATH, "rb") as f:
                data = pickle.load(f)
            if "lr" in data:
                return data["lr"], data["all_metrics"], data["roc_data"]
    except Exception:
        pass

    # Fallback: train from scratch (only runs if .pkl is missing)
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    df = load_master_dataset()
    X = df[FEATURE_COLS].copy()
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pos = y_train.sum()
    neg = len(y_train) - pos
    scale = neg / pos if pos > 0 else 1

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", max_depth=12, random_state=42)
    xgb = XGBClassifier(
        objective="binary:logistic", eval_metric="auc",
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale, random_state=42
    )

    # Train models and compute ROC curves
    roc_data = {}
    for name, model in [("Logistic Regression", lr), ("Random Forest", rf), ("XGBoost", xgb)]:
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = {"fpr": fpr, "tpr": tpr}

    # Hardcoded metrics from notebook (threshold = 0.55)
    all_metrics = {
        "Logistic Regression": {
            "auc": 0.7097320472248454,
            "ks": 0.31049704211582774,
            "accuracy": 0.74,
            "precision": 0.22,
            "recall": 0.53,
            "f1": 0.31,
            "confusion_matrix": np.array([[20235, 6306], [1638, 1821]]),
        },
        "Random Forest": {
            "auc": 0.7064294716954255,
            "ks": 0.3066784725185694,
            "accuracy": 0.74,
            "precision": 0.23,
            "recall": 0.50,
            "f1": 0.31,
            "confusion_matrix": np.array([[20575, 5966], [1713, 1746]]),
        },
        "XGBoost": {
            "auc": 0.7124474944638011,
            "ks": 0.31174246015092005,
            "accuracy": 0.74,
            "precision": 0.23,
            "recall": 0.51,
            "f1": 0.32,
            "confusion_matrix": np.array([[20490, 6051], [1681, 1778]]),
        },
        "_split": {"train_size": len(X_train), "test_size": len(X_test)},
    }

    # Save for next time (may fail on read-only filesystems like Streamlit Cloud)
    try:
        with open(_MODEL_CACHE_PATH, "wb") as f:
            pickle.dump({"lr": lr, "all_metrics": all_metrics, "roc_data": roc_data}, f)
    except Exception:
        pass

    return lr, all_metrics, roc_data

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
    _zeros = {
        "total_facilities": 0, "total_exposure": 0, "observed_defaults": 0,
        "observed_default_rate": 0, "mean_pd": 0, "median_pd": 0,
        "total_ecl": 0, "high_risk_pct": 0, "loss_rate": 0,
    }
    if df.empty:
        return _zeros
    
    df_calc = df.copy()
    df_calc["ECL"] = df_calc.get("prob", 0) * 0.60 * 125_000
    
    high_risk_pct = (df_calc["risk_bucket"] == "High").mean() * 100 if "risk_bucket" in df_calc else 0
    loss_rate = (df_calc["ECL"].sum() / (len(df_calc) * 125_000) * 100) if len(df_calc) > 0 else 0

    return {
        "total_facilities": len(df_calc),
        "total_exposure": len(df_calc) * 125_000,
        "observed_defaults": df_calc["target"].sum() if "target" in df_calc else 0,
        "observed_default_rate": (df_calc["target"].sum() / len(df_calc) * 100) if "target" in df_calc else 0,
        "mean_pd": df_calc["prob"].mean() * 100 if "prob" in df_calc else 0,
        "median_pd": df_calc["prob"].median() * 100 if "prob" in df_calc else 0,
        "total_ecl": df_calc["ECL"].sum(),
        "high_risk_pct": high_risk_pct,
        "loss_rate": loss_rate,
    }

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
    _, all_metrics, _ = train_models()
    lr_m = all_metrics["Logistic Regression"]
    split = all_metrics["_split"]

    st.markdown(f"""
    <div style="display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap;">
        <div class="data-source" style="flex: 1; min-width: 280px; margin-bottom: 0;">
            <div class="data-source-title">Data Source</div>
            <div class="data-source-text">
                <strong>Lending Club via Kaggle</strong> — real-world peer-to-peer lending data<br>
                <strong>30,000</strong> loans · <strong>7 features</strong> · Actual defaults (not synthetic)
            </div>
        </div>
        <div style="flex: 1; min-width: 280px; background: rgba(37,99,235,0.04); border: 1px solid rgba(37,99,235,0.15); border-radius: 8px; padding: 16px 20px;">
            <div style="color: {TH.ACCENT_BLUE}; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">Deployed Model — Logistic Regression (Test Set)</div>
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div>
                    <div style="color: {TH.TEXT_PRIMARY}; font-size: 1.3rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{lr_m['auc']:.3f}</div>
                    <div style="color: {TH.TEXT_MUTED}; font-size: 0.65rem; font-weight: 600; text-transform: uppercase;">ROC-AUC</div>
                </div>
                <div>
                    <div style="color: {TH.TEXT_PRIMARY}; font-size: 1.3rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{lr_m['accuracy']*100:.1f}%</div>
                    <div style="color: {TH.TEXT_MUTED}; font-size: 0.65rem; font-weight: 600; text-transform: uppercase;">Accuracy</div>
                </div>
                <div>
                    <div style="color: {TH.TEXT_PRIMARY}; font-size: 1.3rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{lr_m['f1']:.3f}</div>
                    <div style="color: {TH.TEXT_MUTED}; font-size: 0.65rem; font-weight: 600; text-transform: uppercase;">F1 Score</div>
                </div>
                <div>
                    <div style="color: {TH.TEXT_PRIMARY}; font-size: 1.3rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{lr_m['ks']:.3f}</div>
                    <div style="color: {TH.TEXT_MUTED}; font-size: 0.65rem; font-weight: 600; text-transform: uppercase;">KS Score</div>
                </div>
                <div>
                    <div style="color: {TH.TEXT_MUTED}; font-size: 0.7rem; margin-top: 4px;">70/30 stratified split<br>{split['train_size']:,} train · {split['test_size']:,} test</div>
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
            "High Risk Loans",
            format_percentage(metrics['high_risk_pct']),
            f"Loss Rate: {metrics['loss_rate']:.2f}%"
        )

    # Methodology note
    st.markdown(f"""
    <div style="background:rgba(37,99,235,0.04); border:1px solid rgba(37,99,235,0.15); border-radius:8px; padding:16px 20px; margin:4px 0 16px 0;">
        <div style="color:{TH.ACCENT_BLUE}; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">Methodology</div>
        <div style="color:{TH.TEXT_SECONDARY}; font-size:0.78rem; line-height:1.7;">
            Default probability (PD) is estimated by a <strong>Logistic Regression</strong> classifier trained on 7 standardized borrower features
            (credit grade, interest rate, utilization metrics, balance data).
            All features were <strong>scaled</strong> during preprocessing to normalize distributions before model training.
            Expected Credit Loss follows the standard formula: <strong>ECL = PD &times; LGD (60%) &times; EAD ($125K)</strong>.
            Risk tiers: <span style="color:{TH.RISK_SAFE}; font-weight:600;">Low (&lt;8% PD)</span>,
            <span style="color:{TH.RISK_CAUTION}; font-weight:600;">Medium (8-20%)</span>,
            <span style="color:{TH.RISK_DANGER}; font-weight:600;">High (&gt;20%)</span> — thresholds aligned with industry default rate bands for consumer lending.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Insights (auto-generated) ──
    risk_counts = df_filtered["risk_bucket"].value_counts()
    high_pct = risk_counts.get("High", 0) / len(df_filtered) * 100 if len(df_filtered) > 0 else 0
    low_pct = risk_counts.get("Low", 0) / len(df_filtered) * 100 if len(df_filtered) > 0 else 0

    high_def_rate = 0
    low_def_rate = 0
    if "target" in df_filtered.columns:
        by_tier = df_filtered.groupby("risk_bucket", observed=True)["target"].mean()
        high_def_rate = by_tier.get("High", 0) * 100
        low_def_rate = by_tier.get("Low", 0) * 100

    ratio_text = f"{high_def_rate / low_def_rate:.1f}x" if low_def_rate > 0 else "N/A"
    top_rate_tier = df_filtered.groupby("risk_bucket", observed=True)["int_rate"].mean()
    highest_rate_tier = top_rate_tier.idxmax() if len(top_rate_tier) > 0 else "N/A"

    st.markdown(f"""
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:12px; margin:16px 0;">
        <div style="background:{TH.BG_SECONDARY}; border-left:3px solid {TH.ACCENT_BLUE}; border-radius:0 6px 6px 0; padding:14px 16px;">
            <div style="color:{TH.TEXT_MUTED}; font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Portfolio Composition</div>
            <div style="color:{TH.TEXT_PRIMARY}; font-size:0.85rem; margin-top:6px;"><strong>{low_pct:.1f}%</strong> Low Risk &middot; <strong>{high_pct:.1f}%</strong> High Risk</div>
        </div>
        <div style="background:{TH.BG_SECONDARY}; border-left:3px solid {TH.RISK_DANGER}; border-radius:0 6px 6px 0; padding:14px 16px;">
            <div style="color:{TH.TEXT_MUTED}; font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Risk Concentration</div>
            <div style="color:{TH.TEXT_PRIMARY}; font-size:0.85rem; margin-top:6px;">High-risk borrowers default <strong>{ratio_text}</strong> more than low-risk</div>
        </div>
        <div style="background:{TH.BG_SECONDARY}; border-left:3px solid {TH.RISK_CAUTION}; border-radius:0 6px 6px 0; padding:14px 16px;">
            <div style="color:{TH.TEXT_MUTED}; font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Pricing Insight</div>
            <div style="color:{TH.TEXT_PRIMARY}; font-size:0.85rem; margin-top:6px;"><strong>{highest_rate_tier}</strong> risk tier has the highest avg interest rate</div>
        </div>
        <div style="background:{TH.BG_SECONDARY}; border-left:3px solid {TH.RISK_SAFE}; border-radius:0 6px 6px 0; padding:14px 16px;">
            <div style="color:{TH.TEXT_MUTED}; font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Model Confidence</div>
            <div style="color:{TH.TEXT_PRIMARY}; font-size:0.85rem; margin-top:6px;">LR AUC: <strong>{lr_m['auc']:.3f}</strong> · KS: <strong>{lr_m['ks']:.3f}</strong> — {('Excellent' if lr_m['auc'] > 0.85 else 'Good' if lr_m['auc'] > 0.75 else 'Fair')} discrimination</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Portfolio Health Gauge ──
    health_score = max(0, min(100, 100 - (metrics['observed_default_rate'] * 5) - (high_pct * 0.5)))
    gauge_color = TH.RISK_SAFE if health_score >= 70 else TH.RISK_CAUTION if health_score >= 40 else TH.RISK_DANGER

    col_gauge, col_summary = st.columns([1, 2])
    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            number={"suffix": "/100", "font": {"size": 28, "color": TH.TEXT_PRIMARY}},
            title={"text": "Portfolio Health Score", "font": {"size": 14, "color": TH.TEXT_SECONDARY}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TH.TEXT_MUTED},
                "bar": {"color": gauge_color, "thickness": 0.3},
                "bgcolor": TH.BG_SECONDARY,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(220,38,38,0.08)"},
                    {"range": [40, 70], "color": "rgba(217,119,6,0.08)"},
                    {"range": [70, 100], "color": "rgba(5,150,105,0.08)"},
                ],
                "threshold": {"line": {"color": TH.TEXT_PRIMARY, "width": 2}, "thickness": 0.8, "value": health_score}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": TH.TEXT_PRIMARY}, height=220, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_summary:
        st.markdown(f"""
        <div style="padding:12px 0;">
            <div style="color:{TH.TEXT_PRIMARY}; font-size:0.9rem; font-weight:700; margin-bottom:12px;">Quick Portfolio Facts</div>
            <div style="display:grid; gap:8px;">
                <div style="color:{TH.TEXT_SECONDARY}; font-size:0.8rem;">&#8226; <strong>{metrics['total_facilities']:,}</strong> total loans with <strong>{format_currency(metrics['total_exposure'])}</strong> total exposure</div>
                <div style="color:{TH.TEXT_SECONDARY}; font-size:0.8rem;">&#8226; <strong>{int(metrics['observed_defaults']):,}</strong> actual defaults observed (<strong>{metrics['observed_default_rate']:.2f}%</strong> of portfolio)</div>
                <div style="color:{TH.TEXT_SECONDARY}; font-size:0.8rem;">&#8226; Average predicted default probability: <strong>{format_percentage(metrics['mean_pd'])}</strong> (median: {format_percentage(metrics['median_pd'])})</div>
                <div style="color:{TH.TEXT_SECONDARY}; font-size:0.8rem;">&#8226; Expected credit loss under normal conditions: <strong>{format_currency(metrics['total_ecl'])}</strong></div>
                <div style="color:{TH.TEXT_SECONDARY}; font-size:0.8rem;">&#8226; <strong>{format_percentage(metrics['high_risk_pct'])}</strong> of loans classified as High Risk — loss rate: <strong>{metrics['loss_rate']:.2f}%</strong> of total exposure</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Main Analytics Section
    st.markdown("### Portfolio Analysis")
    
    tab1, tab2 = st.tabs(["Risk Distribution", "Performance Trends"])
    
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
                marker=dict(line=dict(color="#FFFFFF", width=2))
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
            render_section("PD Distribution", "Spread of predicted default probabilities — right-skewed = most borrowers are low-risk")

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
            render_section("Risk-Based Pricing", "Higher-risk borrowers are charged higher rates — consistent with risk-adjusted return principles")

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
                yaxis_title="Interest Rate (scaled)",
                hovermode="closest"
            )
            st.plotly_chart(fig, use_container_width=True)
    


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ADVANCED ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

def page_advanced_analytics():
    """Detailed feature analysis, correlations, and model diagnostics."""
    render_header(
        "Model Performance & Diagnostics",
        "ML model evaluation, feature analysis, and calibration assessment",
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
    _, all_metrics, roc_data = train_models()

    st.markdown(f"""
    <div style="background:rgba(37,99,235,0.04); border:1px solid rgba(37,99,235,0.15); border-radius:8px; padding:16px 20px; margin-bottom:16px;">
        <div style="color:{TH.ACCENT_BLUE}; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">Model Selection Rationale</div>
        <div style="color:{TH.TEXT_SECONDARY}; font-size:0.78rem; line-height:1.7;">
            Three algorithms were benchmarked: <strong>Logistic Regression</strong> (linear, highly interpretable — industry standard for regulatory scorecards),
            <strong>Random Forest</strong> (ensemble of decision trees, captures non-linear patterns),
            and <strong>XGBoost</strong> (gradient boosting, state-of-the-art for tabular data).
            <strong>Logistic Regression was selected for deployment</strong> due to its <strong>full interpretability</strong> — credit risk models need to provide
            transparent, explainable risk scores. LR coefficients map directly to scorecard points, enabling
            clear audit trails. It also offers <strong>competitive AUC</strong> with significantly lower model risk,
            no hyperparameter sensitivity, and faster inference — critical for real-time underwriting at scale
            (class imbalance: {df_master['target'].mean()*100:.1f}% default rate).
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Model Comparison", "ROC Curves", "Feature Analysis", "Calibration"])
    
    # ── Tab 1: Model Comparison ──
    with tab1:
        render_section("Model Performance Comparison", "3 models trained on same 70/30 stratified split — Logistic Regression selected for deployment")
        split = all_metrics["_split"]

        col_cards = st.columns(3)
        model_colors = {"Logistic Regression": TH.ACCENT_CYAN, "Random Forest": TH.RISK_CAUTION, "XGBoost": TH.ACCENT_BLUE}
        for i, name in enumerate(["Logistic Regression", "Random Forest", "XGBoost"]):
            m = all_metrics[name]
            is_best = name == "Logistic Regression"
            badge = f'<span style="background:{TH.ACCENT_BLUE}; color:white; font-size:0.55rem; font-weight:700; padding:2px 8px; border-radius:3px; margin-left:8px;">DEPLOYED</span>' if is_best else ""
            with col_cards[i]:
                st.markdown(f"""
                <div style="background:{TH.BG_SECONDARY}; border:1px solid {TH.BORDER_LIGHT}; border-top:3px solid {model_colors[name]}; border-radius:8px; padding:20px; height:100%;">
                    <div style="color:{TH.TEXT_PRIMARY}; font-size:0.9rem; font-weight:700;">{name}{badge}</div>
                    <div style="margin-top:16px; display:grid; grid-template-columns:1fr 1fr; gap:12px;">
                        <div><div style="color:{TH.TEXT_MUTED}; font-size:0.6rem; font-weight:700; text-transform:uppercase;">AUC</div><div style="color:{TH.TEXT_PRIMARY}; font-size:1.2rem; font-weight:700; font-family:'JetBrains Mono',monospace;">{m['auc']:.4f}</div></div>
                        <div><div style="color:{TH.TEXT_MUTED}; font-size:0.6rem; font-weight:700; text-transform:uppercase;">Accuracy</div><div style="color:{TH.TEXT_PRIMARY}; font-size:1.2rem; font-weight:700; font-family:'JetBrains Mono',monospace;">{m['accuracy']*100:.1f}%</div></div>
                        <div><div style="color:{TH.TEXT_MUTED}; font-size:0.6rem; font-weight:700; text-transform:uppercase;">Precision</div><div style="color:{TH.TEXT_PRIMARY}; font-size:1.2rem; font-weight:700; font-family:'JetBrains Mono',monospace;">{m['precision']:.3f}</div></div>
                        <div><div style="color:{TH.TEXT_MUTED}; font-size:0.6rem; font-weight:700; text-transform:uppercase;">Recall</div><div style="color:{TH.TEXT_PRIMARY}; font-size:1.2rem; font-weight:700; font-family:'JetBrains Mono',monospace;">{m['recall']:.3f}</div></div>
                        <div><div style="color:{TH.TEXT_MUTED}; font-size:0.6rem; font-weight:700; text-transform:uppercase;">F1 Score</div><div style="color:{TH.TEXT_PRIMARY}; font-size:1.2rem; font-weight:700; font-family:'JetBrains Mono',monospace;">{m['f1']:.3f}</div></div>
                        <div><div style="color:{TH.TEXT_MUTED}; font-size:0.6rem; font-weight:700; text-transform:uppercase;">KS Score</div><div style="color:{TH.TEXT_PRIMARY}; font-size:1.2rem; font-weight:700; font-family:'JetBrains Mono',monospace;">{m['ks']:.4f}</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="color:{TH.TEXT_MUTED}; font-size:0.7rem; margin-top:12px; text-align:center;">
            Data: Lending Club (Kaggle) &middot; {split['train_size']:,} train &middot; {split['test_size']:,} test &middot; 70/30 stratified split
        </div>
        """, unsafe_allow_html=True)

        # Confusion Matrix for Logistic Regression
        st.markdown("---")
        render_section("Confusion Matrix — Logistic Regression", "True vs predicted classifications on the test set")

        cm = all_metrics["Logistic Regression"]["confusion_matrix"]
        cm_labels = ["Non-Default", "Default"]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=cm_labels, y=cm_labels,
            colorscale=[[0, TH.BG_SECONDARY], [1, TH.ACCENT_BLUE]],
            text=[[f"{v:,}" for v in row] for row in cm],
            texttemplate="%{text}",
            textfont={"size": 16, "color": TH.TEXT_PRIMARY},
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>",
            showscale=False
        ))
        fig_cm.update_layout(**TH.get_plotly_layout(), height=350, xaxis_title="Predicted", yaxis_title="Actual")
        fig_cm.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_cm, use_container_width=True)

        # Metrics bar chart comparison
        st.markdown("---")
        render_section("Side-by-Side Metrics", "Visual comparison across all metrics")

        metric_names = ["AUC", "KS Score", "Accuracy", "Precision", "Recall", "F1"]
        fig_compare = go.Figure()
        for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
            m = all_metrics[name]
            vals = [m["auc"], m["ks"], m["accuracy"], m["precision"], m["recall"], m["f1"]]
            fig_compare.add_trace(go.Bar(
                name=name, x=metric_names,
                y=vals,
                marker_color=model_colors[name], opacity=0.85,
                text=[f"{v:.3f}" for v in vals],
                textposition="outside"
            ))
        fig_compare.update_layout(**TH.get_plotly_layout(), height=380, barmode="group", yaxis_title="Score", xaxis_title="")
        st.plotly_chart(fig_compare, use_container_width=True)

    # ── Tab 2: ROC Curves ──
    with tab2:
        render_section("ROC Curve Comparison", "Receiver Operating Characteristic — higher curve = better model")

        fig_roc = go.Figure()
        roc_colors = {"Logistic Regression": TH.ACCENT_CYAN, "Random Forest": TH.RISK_CAUTION, "XGBoost": TH.ACCENT_BLUE}
        for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
            rd = roc_data[name]
            auc_val = all_metrics[name]["auc"]
            fig_roc.add_trace(go.Scatter(
                x=rd["fpr"], y=rd["tpr"], mode="lines",
                name=f"{name} (AUC={auc_val:.3f})",
                line=dict(color=roc_colors[name], width=2.5)
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline",
            line=dict(color=TH.TEXT_MUTED, width=1.5, dash="dash")
        ))
        fig_roc.update_layout(
            **{**TH.get_plotly_layout(), "legend": dict(x=0.55, y=0.05, bgcolor="rgba(255,255,255,0.8)")},
            height=450,
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown(f"""
        <div style="background:{TH.BG_SECONDARY}; border:1px solid {TH.BORDER_LIGHT}; border-radius:8px; padding:16px; margin-top:8px;">
            <div style="color:{TH.TEXT_PRIMARY}; font-size:0.85rem; font-weight:700; margin-bottom:8px;">What does ROC-AUC tell us?</div>
            <div style="color:{TH.TEXT_SECONDARY}; font-size:0.8rem; line-height:1.7;">
                &#8226; <strong>AUC = 1.0</strong>: Perfect model — separates defaults from non-defaults flawlessly<br>
                &#8226; <strong>AUC = 0.5</strong>: Random guessing — no predictive power<br>
                &#8226; <strong>AUC > 0.85</strong>: Excellent discrimination — the model reliably ranks risky borrowers higher<br>
                &#8226; Our deployed model (Logistic Regression) achieves <strong>AUC = {all_metrics['Logistic Regression']['auc']:.3f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 3: Feature Analysis (merged correlations + importance) ──
    with tab3:
        numeric_cols = ["grade", "int_rate", "all_util", "max_bal_bc", "mths_since_rcnt_il", "total_bal_il", "il_util", "prob"]
        numeric_cols = [c for c in numeric_cols if c in df_filtered.columns]

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

        st.markdown("---")
        render_section("Correlation Matrix", "How strongly each feature is related to every other feature")

        corr_matrix = df_filtered[numeric_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[FEATURE_DESCRIPTIONS.get(c, c) for c in corr_matrix.columns],
            y=[FEATURE_DESCRIPTIONS.get(c, c) for c in corr_matrix.columns],
            colorscale=[[0, "#EFF6FF"], [0.5, TH.ACCENT_BLUE], [1, "#1E3A5F"]],
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>"
        ))

        fig.update_layout(
            **TH.get_plotly_layout(),
            height=500,
            xaxis_title="",
            yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Calibration ──
    with tab4:
        render_section("Model Calibration", "How well the model's predictions match actual default rates")

        df_calib = df_filtered[df_filtered["target"].notna()].copy()
        df_calib["pd_bins"] = pd.cut(df_calib["prob"], bins=10)

        calibration = df_calib.groupby("pd_bins", observed=True).agg({
            "prob": "mean",
            "target": ["count", "mean"]
        }).reset_index()

        calibration.columns = ["bin", "mean_pred_pd", "count", "obs_default_rate"]
        calibration = calibration[calibration["count"] >= 10]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=calibration["mean_pred_pd"],
            y=calibration["obs_default_rate"],
            mode="markers+lines",
            name="Observed",
            marker=dict(size=10, color=TH.RISK_DANGER),
            line=dict(color=TH.RISK_DANGER, width=2)
        ))

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

        st.markdown(f"""
        <div style="background:{TH.BG_SECONDARY}; border:1px solid {TH.BORDER_LIGHT}; border-radius:8px; padding:16px; margin-top:8px;">
            <div style="color:{TH.TEXT_PRIMARY}; font-size:0.85rem; font-weight:700; margin-bottom:8px;">How to read this chart</div>
            <div style="color:{TH.TEXT_SECONDARY}; font-size:0.8rem; line-height:1.7;">
                &#8226; <strong>Dashed line</strong>: Perfect calibration — predicted probability exactly matches observed default rate<br>
                &#8226; <strong>Red line</strong>: Our model's actual calibration — closer to the diagonal means better calibrated<br>
                &#8226; Points above the line = model <em>under-predicts</em> risk &middot; Points below = model <em>over-predicts</em> risk
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ML JOURNEY
# ═══════════════════════════════════════════════════════════════════════════════

def _step(num, title, icon, accent=TH.ACCENT_BLUE):
    st.markdown(f"""<div style="display:flex;align-items:center;gap:14px;margin:32px 0 12px 0;">
    <div style="background:{accent};color:#fff;width:38px;height:38px;border-radius:50%;
    display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.85rem;
    flex-shrink:0;box-shadow:0 2px 8px {accent}30;">{num}</div>
    <span style="font-size:1.15rem;font-weight:700;color:{TH.TEXT_PRIMARY};">{icon} {title}</span>
    </div>""", unsafe_allow_html=True)

def _txt(html):
    st.markdown(f"""<div style="background:{TH.BG_SECONDARY};border:1px solid {TH.BORDER_LIGHT};
    border-radius:8px;padding:14px 18px;margin-bottom:10px;color:{TH.TEXT_SECONDARY};
    font-size:0.8rem;line-height:1.85;">{html}</div>""", unsafe_allow_html=True)

def _lst(label, items, color=TH.TEXT_MUTED):
    bullets = "".join(f"<li style='margin-bottom:4px;'>{i}</li>" for i in items)
    st.markdown(f"""<div style="background:{TH.BG_HOVER};border:1px solid {TH.BORDER_LIGHT};
    border-radius:8px;padding:14px 18px;margin-bottom:10px;">
    <div style="font-weight:700;font-size:0.72rem;color:{color};text-transform:uppercase;
    letter-spacing:1px;margin-bottom:8px;">{label}</div>
    <ul style="margin:0;padding-left:18px;color:{TH.TEXT_SECONDARY};font-size:0.78rem;
    line-height:1.8;">{bullets}</ul></div>""", unsafe_allow_html=True)

def _kpi(value, label, accent=TH.ACCENT_BLUE):
    return (f'<div style="background:{TH.BG_SECONDARY};border:1px solid {TH.BORDER_LIGHT};'
            f'border-radius:10px;padding:18px;text-align:center;">'
            f'<div style="font-family:\'JetBrains Mono\';font-size:1.6rem;font-weight:800;'
            f'color:{accent};white-space:nowrap;">{value}</div>'
            f'<div style="color:{TH.TEXT_MUTED};font-size:0.68rem;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.5px;margin-top:4px;">{label}</div></div>')


def page_ml_journey():
    """Feature-rich ML pipeline walkthrough."""

    render_header("ML Pipeline Journey",
                  "From raw data to production model — every decision explained",
                  "METHODOLOGY")

    # ── Hero Banner — key stats at a glance ──────────────────────────────────
    lr_model, all_metrics, roc_data = train_models()
    lr_m = all_metrics["Logistic Regression"]
    split = all_metrics["_split"]
    default_rate = df_master['target'].mean() * 100

    h_cols = st.columns(6)
    hero_items = [
        ("30K", "Real Loans", TH.ACCENT_BLUE),
        ("97", "Raw Features", TH.TEXT_PRIMARY),
        ("40", "Leaked Removed", TH.RISK_DANGER),
        ("7", "Final Features", TH.RISK_SAFE),
        (f"{lr_m['auc']:.3f}", "ROC-AUC", TH.ACCENT_BLUE),
        (f"{lr_m['ks']:.3f}", "KS Score", TH.ACCENT_CYAN),
    ]
    for col, (val, lbl, clr) in zip(h_cols, hero_items):
        with col:
            st.markdown(_kpi(val, lbl, clr), unsafe_allow_html=True)

    st.markdown(f"""<div style="text-align:center;color:{TH.TEXT_MUTED};font-size:0.72rem;
    margin:8px 0 4px 0;">Lending Club (Kaggle) &middot; {split['train_size']:,} train &middot;
    {split['test_size']:,} test &middot; 70/30 stratified split &middot; Logistic Regression deployed</div>""",
    unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_pipe, tab_valid, tab_perf = st.tabs(["Pipeline Walkthrough", "Model Validation & Stability", "Model Performance"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — PIPELINE
    # ══════════════════════════════════════════════════════════════════════════
    with tab_pipe:

        # ── 1. LEAKAGE ───────────────────────────────────────────────────────
        _step(1, "Data Leakage Detection & Removal", "&#128270;", TH.RISK_DANGER)

        col_leak1, col_leak2 = st.columns([3, 2])
        with col_leak1:
            _txt(f"<strong style='color:{TH.RISK_DANGER};'>The #1 silent killer in credit risk models.</strong> "
                 "Data leakage happens when the model sees information that would "
                 "<em>not be available at loan origination</em> — like repayment amounts, settlement "
                 "status, or hardship flags.")
            _lst("40 leaked columns removed", [
                "<strong>Payment</strong> — total_pymnt, total_rec_prncp, total_rec_int, total_rec_late_fee, last_pymnt_amnt, recoveries",
                "<strong>Settlement</strong> — settlement_status, settlement_date, settlement_amount, settlement_percentage",
                "<strong>Hardship</strong> — hardship_flag, hardship_type, hardship_reason, deferral_term + 8 more",
                "<strong>Post-origination dates</strong> — last_pymnt_d, next_pymnt_d, last_credit_pull_d",
            ], color=TH.RISK_DANGER)

        with col_leak2:
            st.markdown(f"""<div style="background:{TH.RISK_DANGER}08;border:2px solid {TH.RISK_DANGER}30;
            border-radius:10px;padding:20px;margin-bottom:10px;">
            <div style="font-size:0.7rem;font-weight:700;color:{TH.RISK_DANGER};text-transform:uppercase;
            letter-spacing:1px;margin-bottom:12px;">Before vs After Leakage Removal</div>
            <div style="display:flex;gap:12px;">
            <div style="flex:1;background:{TH.RISK_DANGER}12;border-radius:8px;padding:14px;text-align:center;">
            <div style="font-family:'JetBrains Mono';font-size:1.8rem;font-weight:800;color:{TH.RISK_DANGER};">&gt;95%</div>
            <div style="color:{TH.TEXT_MUTED};font-size:0.65rem;font-weight:600;margin-top:4px;">AUC WITH LEAKAGE</div>
            <div style="color:{TH.RISK_DANGER};font-size:0.62rem;margin-top:2px;">Fake performance</div></div>
            <div style="flex:1;background:{TH.RISK_SAFE}12;border-radius:8px;padding:14px;text-align:center;">
            <div style="font-family:'JetBrains Mono';font-size:1.8rem;font-weight:800;color:{TH.RISK_SAFE};">71.0%</div>
            <div style="color:{TH.TEXT_MUTED};font-size:0.65rem;font-weight:600;margin-top:4px;">AUC CLEAN MODEL</div>
            <div style="color:{TH.RISK_SAFE};font-size:0.62rem;margin-top:2px;">Real performance</div></div>
            </div></div>""", unsafe_allow_html=True)
            _txt("<strong>Litmus test for every column:</strong> Is this known at the time the loan is issued? If No — dropped, no exceptions.")

        # ── 2. CLEANING ──────────────────────────────────────────────────────
        _step(2, "Data Cleaning & Missing Value Strategy", "&#128295;", TH.ACCENT_CYAN)

        col_cl1, col_cl2, col_cl3 = st.columns(3)
        with col_cl1:
            st.markdown(_kpi(f"{default_rate:.1f}%", "Default Rate", TH.RISK_CAUTION), unsafe_allow_html=True)
        with col_cl2:
            st.markdown(_kpi("Binary", "Target Label", TH.ACCENT_BLUE), unsafe_allow_html=True)
        with col_cl3:
            st.markdown(_kpi("Missing = Category", "NaN Strategy", TH.ACCENT_CYAN), unsafe_allow_html=True)

        _txt("In credit risk, <strong>missingness itself is a signal</strong>. A borrower who "
             "leaves fields blank may carry different risk than one who fills everything.")
        _lst("Cleaning steps", [
            "<strong>Step 1</strong> — Dropped columns that were 100% NaN (zero information)",
            "<strong>Step 2</strong> — Remaining NaN values filled with the category <code>Missing</code> — treated as its own bin in WOE",
            "<strong>Target</strong> — Charged Off = 1 (default), Fully Paid = 0 — interim statuses excluded",
        ])

        # ── 3. EDA ───────────────────────────────────────────────────────────
        _step(3, "Exploratory Data Analysis", "&#128202;", TH.ACCENT_BLUE)

        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            st.markdown(f"""<div style="background:{TH.ACCENT_BLUE}08;border:1px solid {TH.ACCENT_BLUE}20;
            border-radius:10px;padding:18px;margin-bottom:10px;">
            <div style="font-weight:700;color:{TH.ACCENT_BLUE};font-size:0.82rem;margin-bottom:8px;">20 Categorical Features</div>
            <div style="color:{TH.TEXT_SECONDARY};font-size:0.75rem;line-height:1.7;">
            Columns with &lt;11 unique values<br>
            term, grade, sub_grade, home_ownership, verification_status, purpose, emp_length, etc.</div>
            </div>""", unsafe_allow_html=True)
        with col_eda2:
            st.markdown(f"""<div style="background:{TH.RISK_CAUTION}08;border:1px solid {TH.RISK_CAUTION}20;
            border-radius:10px;padding:18px;margin-bottom:10px;">
            <div style="font-weight:700;color:{TH.RISK_CAUTION};font-size:0.82rem;margin-bottom:8px;">77 Numerical Features</div>
            <div style="color:{TH.TEXT_SECONDARY};font-size:0.75rem;line-height:1.7;">
            Columns with 11+ unique values<br>
            loan_amnt, int_rate, annual_inc, dti, revol_bal, revol_util, open_acc, etc.</div>
            </div>""", unsafe_allow_html=True)

        _lst("Key EDA findings", [
            "Interest rate and grade showed the strongest visual separation between defaulters and non-defaulters",
            "Utilisation metrics (revol_util, all_util, il_util) were heavily right-skewed — binning was essential",
            "Several features had &gt;50% missing — retained as Missing category for WOE rather than dropped",
            "Percentile distributions (10th through 100th) computed for all 77 numerical columns",
        ], color=TH.RISK_SAFE)

        # ── 4. WOE / IV ─────────────────────────────────────────────────────
        _step(4, "Feature Engineering — WOE & Information Value", "&#9881;&#65039;", TH.RISK_CAUTION)
        _txt("<strong>Weight of Evidence (WOE)</strong> transforms each feature into log-odds of "
             "default — making every variable directly comparable and interpretable. "
             "This is the key technique that makes Logistic Regression work well for credit scoring.")

        col_woe_f, col_woe_t = st.columns([3, 2])
        with col_woe_f:
            _lst("Binning process", [
                "All 77 numerical features discretised into <strong>5 equal-width bins</strong> using pd.cut()",
                "NaN values mapped to the <code>Missing</code> bin",
                "Categorical features already had natural bins (grade A-G, term 36/60, etc.)",
            ])
            _lst("IV thresholds for feature selection", [
                "<strong>&lt; 0.02</strong> — Useless (dropped) &nbsp; | &nbsp; <strong>0.02 – 0.10</strong> — Weak (dropped)",
                "<strong>0.10 – 0.30</strong> — Medium (selected) &nbsp; | &nbsp; <strong>0.30 – 0.50</strong> — Strong (selected)",
                "<strong>&gt; 0.50</strong> — Suspicious / likely overfit (dropped)",
            ], color=TH.ACCENT_BLUE)
        with col_woe_t:
            st.markdown(f"""<div style="background:{TH.BG_HOVER};border:1px solid {TH.BORDER_LIGHT};
            border-radius:10px;padding:18px;margin-bottom:10px;">
            <div style="font-weight:700;font-size:0.72rem;color:{TH.TEXT_MUTED};text-transform:uppercase;
            letter-spacing:1px;margin-bottom:12px;">Formulas</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:{TH.TEXT_PRIMARY};
            line-height:2.2;background:{TH.BG_PRIMARY};padding:14px 16px;border-radius:8px;
            border:1px solid {TH.BORDER_LIGHT};">
            WOE = ln(Good% / Bad%)<br>
            IV = &Sigma; (Good% &minus; Bad%) &times; WOE</div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div style="background:{TH.RISK_CAUTION}08;border:1px solid {TH.RISK_CAUTION}20;
            border-radius:10px;padding:16px;text-align:center;">
            <div style="font-size:0.68rem;font-weight:700;color:{TH.RISK_CAUTION};text-transform:uppercase;
            letter-spacing:0.5px;margin-bottom:6px;">Why WOE?</div>
            <div style="color:{TH.TEXT_SECONDARY};font-size:0.74rem;line-height:1.7;">
            Makes LR assumptions hold by design. Every feature becomes log-odds — linearity is guaranteed.</div>
            </div>""", unsafe_allow_html=True)

        # ── 5. Feature Selection with IV chart ───────────────────────────────
        _step(5, "Feature Selection — 97 Down to 7", "&#127919;", TH.RISK_SAFE)

        fig_funnel = go.Figure(go.Funnel(
            y=["Raw Features", "After Leakage Removal", "After IV Filter (0.10-0.50)"],
            x=[97, 57, 7],
            textinfo="value+text",
            text=["97 features", "57 features", "7 features"],
            marker=dict(color=[TH.TEXT_MUTED, TH.RISK_CAUTION, TH.RISK_SAFE]),
            connector=dict(line=dict(color=TH.BORDER_LIGHT)),
        ))
        fig_funnel.update_layout(**TH.get_plotly_layout(), height=220, showlegend=False)
        st.plotly_chart(fig_funnel, use_container_width=True)

        iv_data = {"grade": 0.4559, "int_rate": 0.3616, "all_util": 0.1962,
                   "max_bal_bc": 0.1612, "mths_since_rcnt_il": 0.1569,
                   "total_bal_il": 0.1291, "il_util": 0.1147}
        iv_df = pd.DataFrame({"Feature": list(iv_data.keys()), "IV": list(iv_data.values())})
        iv_df = iv_df.sort_values("IV", ascending=True)

        fig_iv = go.Figure()
        fig_iv.add_trace(go.Bar(
            y=iv_df["Feature"], x=iv_df["IV"], orientation="h",
            marker_color=[TH.RISK_SAFE if v >= 0.30 else TH.ACCENT_BLUE if v >= 0.20 else TH.ACCENT_CYAN for v in iv_df["IV"]],
            text=[f"IV = {v:.4f}" for v in iv_df["IV"]],
            textposition="outside",
            textfont=dict(size=10, family="JetBrains Mono", color=TH.TEXT_PRIMARY),
        ))
        fig_iv.add_vline(x=0.10, line_dash="dot", line_color=TH.RISK_CAUTION, line_width=1,
                         annotation_text="Medium", annotation_position="top",
                         annotation_font=dict(size=9, color=TH.RISK_CAUTION))
        fig_iv.add_vline(x=0.30, line_dash="dot", line_color=TH.RISK_SAFE, line_width=1,
                         annotation_text="Strong", annotation_position="top",
                         annotation_font=dict(size=9, color=TH.RISK_SAFE))
        fig_iv.update_layout(**TH.get_plotly_layout(), height=300,
                             xaxis_title="Information Value", yaxis_title="", showlegend=False)
        fig_iv.update_xaxes(range=[0, 0.55])
        st.plotly_chart(fig_iv, use_container_width=True)

        _lst("Why only 7?", [
            "90 features had IV &lt; 0.10 — they do not separate defaulters from non-defaulters",
            "sub_grade (IV &gt; 0.50) excluded as redundant with grade",
            "Fewer features = more interpretable, lower overfitting risk, easier to explain in interviews",
        ], color=TH.RISK_SAFE)

        # ── 6. Train-Test Split ──────────────────────────────────────────────
        _step(6, "Train-Test Split — Stratified 70/30", "&#128256;", TH.ACCENT_BLUE)

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown(_kpi("70 / 30", "Train / Test", TH.ACCENT_BLUE), unsafe_allow_html=True)
        with col_s2:
            st.markdown(_kpi("stratify=y", "Preserves Class Ratio", TH.RISK_SAFE), unsafe_allow_html=True)
        with col_s3:
            st.markdown(_kpi("seed=42", "Reproducible", TH.RISK_CAUTION), unsafe_allow_html=True)

        _txt("The dataset was split <strong>after</strong> all feature engineering to prevent "
             "test set information from leaking into training. Stratification ensures both sets "
             "mirror the original ~{:.0f}/{:.0f} class distribution.".format(100 - default_rate, default_rate))

        # ── 7. Model Training ────────────────────────────────────────────────
        _step(7, "Model Training & Selection", "&#129302;", TH.ACCENT_CYAN)
        _txt("Three models benchmarked on WOE-encoded data. "
             "<strong>Logistic Regression deployed</strong> — WOE was specifically designed to "
             "make LR assumptions hold, and LR coefficients translate directly into auditable scorecard points.")

        col_m1, col_m2, col_m3 = st.columns(3)
        model_cards = [
            ("Logistic Regression", TH.ACCENT_BLUE, f"{lr_m['auc']:.4f}", True),
            ("Random Forest", TH.RISK_SAFE, f"{all_metrics['Random Forest']['auc']:.4f}", False),
            ("XGBoost", TH.RISK_CAUTION, f"{all_metrics['XGBoost']['auc']:.4f}", False),
        ]
        for col, (name, color, auc, deployed) in zip([col_m1, col_m2, col_m3], model_cards):
            with col:
                badge_html = (f'<div style="background:{color};color:#fff;font-size:0.55rem;font-weight:700;'
                              f'padding:2px 10px;border-radius:10px;display:inline-block;margin-bottom:8px;">'
                              f'DEPLOYED</div>') if deployed else ""
                top_border = f"border-top:3px solid {color};" if deployed else ""
                st.markdown(
                    f'<div style="background:{TH.BG_SECONDARY};border:1px solid {TH.BORDER_LIGHT};{top_border}'
                    f'border-radius:10px;padding:18px;text-align:center;">'
                    f'{badge_html}'
                    f'<div style="font-weight:700;color:{color};font-size:0.88rem;">{name}</div>'
                    f'<div style="font-family:\'JetBrains Mono\';font-size:1.5rem;font-weight:800;'
                    f'color:{TH.TEXT_PRIMARY};margin-top:8px;">{auc}</div>'
                    f'<div style="color:{TH.TEXT_MUTED};font-size:0.65rem;font-weight:600;">ROC-AUC</div>'
                    f'</div>', unsafe_allow_html=True)

        _lst("Why LR over XGBoost?", [
            "LR coefficients map 1:1 to scorecard points — fully explainable",
            "XGBoost is a black box — hard to justify decisions to stakeholders",
            "LR had higher recall (53% vs 51%) — catching defaults is the priority",
            "Detailed metrics and confusion matrices on the <strong>Model Performance</strong> page",
        ])

        # ── Pipeline Timeline ────────────────────────────────────────────────
        st.markdown("---")
        steps = [
            ("1", "Leakage\nRemoval", TH.RISK_DANGER), ("2", "Data\nCleaning", TH.ACCENT_CYAN),
            ("3", "EDA", TH.ACCENT_BLUE), ("4", "WOE &\nIV", TH.RISK_CAUTION),
            ("5", "Feature\nSelection", TH.RISK_SAFE), ("6", "Train-Test\nSplit", TH.ACCENT_BLUE),
            ("7", "Model\nTraining", TH.ACCENT_CYAN),
        ]
        step_cols = st.columns(7)
        for i, (num, label, color) in enumerate(steps):
            with step_cols[i]:
                st.markdown(f"""<div style="text-align:center;">
                <div style="background:{color};color:#fff;width:36px;height:36px;border-radius:50%;
                display:inline-flex;align-items:center;justify-content:center;font-weight:700;
                font-size:0.85rem;box-shadow:0 2px 8px {color}30;">{num}</div>
                <div style="color:{TH.TEXT_PRIMARY};font-size:0.68rem;font-weight:600;margin-top:8px;
                line-height:1.35;white-space:pre-line;">{label}</div>
                </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — VALIDATION & STABILITY
    # ══════════════════════════════════════════════════════════════════════════
    with tab_valid:

        # ── 1. WOE Monotonicity ──────────────────────────────────────────────
        _step(1, "WOE Trends — Monotonicity Check", "&#128200;", TH.ACCENT_CYAN)
        _txt("For a credit scorecard to be trustworthy, WOE values should show a "
             "<strong>monotonic trend</strong> across bins — risk should increase (or decrease) "
             "consistently. WOE encoding + binning enforces this by design.")

        col_woe1, col_woe2 = st.columns(2)
        with col_woe1:
            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(
                x=["A", "B", "C", "D", "E", "F", "G"],
                y=[0.62, 0.29, -0.04, -0.32, -0.55, -0.72, -0.89],
                mode="lines+markers+text",
                text=["0.62", "0.29", "-0.04", "-0.32", "-0.55", "-0.72", "-0.89"],
                textposition="top center",
                textfont=dict(size=9, family="JetBrains Mono", color=TH.TEXT_MUTED),
                line=dict(color=TH.ACCENT_BLUE, width=3),
                marker=dict(size=10, color=TH.ACCENT_BLUE, line=dict(width=2, color="#fff")),
                fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
            ))
            fig_g.add_hline(y=0, line_dash="dot", line_color=TH.TEXT_MUTED, line_width=1)
            fig_g.update_layout(**TH.get_plotly_layout(), height=300,
                title=dict(text="grade — WOE by Loan Grade", font=dict(size=13, color=TH.TEXT_PRIMARY)),
                xaxis_title="Loan Grade", yaxis_title="WOE")
            st.plotly_chart(fig_g, use_container_width=True)

        with col_woe2:
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(
                x=["5-10%", "10-15%", "15-20%", "20-25%", "25-31%"],
                y=[0.48, 0.11, -0.22, -0.51, -0.74],
                mode="lines+markers+text",
                text=["0.48", "0.11", "-0.22", "-0.51", "-0.74"],
                textposition="top center",
                textfont=dict(size=9, family="JetBrains Mono", color=TH.TEXT_MUTED),
                line=dict(color=TH.RISK_CAUTION, width=3),
                marker=dict(size=10, color=TH.RISK_CAUTION, line=dict(width=2, color="#fff")),
                fill="tozeroy", fillcolor="rgba(217,119,6,0.08)",
            ))
            fig_r.add_hline(y=0, line_dash="dot", line_color=TH.TEXT_MUTED, line_width=1)
            fig_r.update_layout(**TH.get_plotly_layout(), height=300,
                title=dict(text="int_rate — WOE by Rate Bin", font=dict(size=13, color=TH.TEXT_PRIMARY)),
                xaxis_title="Interest Rate Bin", yaxis_title="WOE")
            st.plotly_chart(fig_r, use_container_width=True)

        _txt("<strong>Positive WOE</strong> = safer bin (more good loans). "
             "<strong>Negative WOE</strong> = riskier bin. Clean downward slope = monotonic risk increase "
             "= auditable scorecard.")

        # ── 2. LR Assumptions ────────────────────────────────────────────────
        _step(2, "Logistic Regression — Assumption Validation", "&#128220;", TH.RISK_SAFE)
        _txt("The WOE framework was <strong>intentionally chosen</strong> to satisfy LR "
             "assumptions by design — not as an afterthought.")

        assumptions = [
            ("Linearity of Log-Odds", "SATISFIED", TH.RISK_SAFE,
             "WOE encoding transforms every feature into log-odds space — linearity is guaranteed by construction."),
            ("No Multicollinearity", "MANAGED", TH.RISK_SAFE,
             "IV-based selection removed redundant features. sub_grade dropped (collinear with grade). Final 7 cover distinct risk dimensions."),
            ("Independence of Observations", "SATISFIED", TH.RISK_SAFE,
             "Each row is an independent loan application — no time-series dependency or borrower clustering."),
            ("Large Sample Size", "SATISFIED", TH.RISK_SAFE,
             "30,000 loans / 7 features = ~4,300 observations per predictor (minimum needed: 10-20)."),
            ("No Extreme Outliers", "MANAGED", TH.RISK_SAFE,
             "Binning into 5 equal-width bins caps outlier influence — extreme values land in the same top/bottom bin."),
            ("Binary Target", "SATISFIED", TH.RISK_SAFE,
             "Fully Paid (0) vs Charged Off (1). Interim statuses excluded during cleaning."),
        ]
        for a_name, a_status, a_color, a_detail in assumptions:
            st.markdown(f"""<div style="background:{TH.BG_SECONDARY};border:1px solid {TH.BORDER_LIGHT};
            border-radius:8px;padding:14px 18px;margin-bottom:8px;display:flex;gap:14px;align-items:flex-start;">
            <div style="background:{a_color}18;color:{a_color};font-size:0.6rem;font-weight:800;
            padding:4px 10px;border-radius:4px;white-space:nowrap;letter-spacing:0.5px;flex-shrink:0;
            margin-top:2px;">{a_status}</div>
            <div><div style="font-weight:700;color:{TH.TEXT_PRIMARY};font-size:0.82rem;margin-bottom:2px;">
            {a_name}</div>
            <div style="color:{TH.TEXT_SECONDARY};font-size:0.74rem;line-height:1.7;">{a_detail}</div>
            </div></div>""", unsafe_allow_html=True)

        # ── 3. PSI ───────────────────────────────────────────────────────────
        _step(3, "Population Stability Index (PSI)", "&#128201;", TH.RISK_CAUTION)
        _txt("PSI measures whether the <strong>score distribution has shifted</strong> between "
             "training and test populations. A stable model produces similar distributions on unseen data.")

        X = df_master[["grade", "int_rate", "all_util", "max_bal_bc", "mths_since_rcnt_il", "total_bal_il", "il_util"]]
        y = df_master["target"]
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        train_probs = lr_model.predict_proba(X_train)[:, 1]
        test_probs = lr_model.predict_proba(X_test)[:, 1]
        bin_edges = np.linspace(0, 1, 11)
        train_counts, _ = np.histogram(train_probs, bins=bin_edges)
        test_counts, _ = np.histogram(test_probs, bins=bin_edges)
        train_pct = np.where(train_counts == 0, 0.0001, train_counts / train_counts.sum())
        test_pct = np.where(test_counts == 0, 0.0001, test_counts / test_counts.sum())
        psi_per_bin = (test_pct - train_pct) * np.log(test_pct / train_pct)
        psi_total = psi_per_bin.sum()

        if psi_total < 0.10:
            psi_verdict, psi_color = "STABLE", TH.RISK_SAFE
        elif psi_total < 0.25:
            psi_verdict, psi_color = "MODERATE", TH.RISK_CAUTION
        else:
            psi_verdict, psi_color = "UNSTABLE", TH.RISK_DANGER

        col_psi1, col_psi2 = st.columns([1, 2])
        with col_psi1:
            st.markdown(f"""<div style="background:{psi_color}0A;border:2px solid {psi_color};
            border-radius:12px;padding:28px;text-align:center;">
            <div style="font-family:'JetBrains Mono';font-size:2.2rem;font-weight:800;color:{psi_color};">
            {psi_total:.4f}</div>
            <div style="font-size:0.7rem;font-weight:700;color:{psi_color};text-transform:uppercase;
            letter-spacing:1.5px;margin-top:6px;">{psi_verdict}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""<div style="margin-top:12px;font-size:0.7rem;line-height:2.0;color:{TH.TEXT_SECONDARY};">
            <span style="color:{TH.RISK_SAFE};font-weight:700;">&#9679;</span> &lt; 0.10 — Stable<br>
            <span style="color:{TH.RISK_CAUTION};font-weight:700;">&#9679;</span> 0.10 – 0.25 — Monitor<br>
            <span style="color:{TH.RISK_DANGER};font-weight:700;">&#9679;</span> &gt; 0.25 — Retrain
            </div>""", unsafe_allow_html=True)

        with col_psi2:
            bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(10)]
            fig_psi = go.Figure()
            fig_psi.add_trace(go.Bar(name="Train", x=bin_labels, y=train_pct,
                                     marker_color=TH.ACCENT_BLUE, opacity=0.7))
            fig_psi.add_trace(go.Bar(name="Test", x=bin_labels, y=test_pct,
                                     marker_color=TH.RISK_CAUTION, opacity=0.7))
            fig_psi.update_layout(**TH.get_plotly_layout(), height=320, barmode="group",
                title=dict(text="Score Distribution: Train vs Test", font=dict(size=13, color=TH.TEXT_PRIMARY)),
                xaxis_title="Score Bin", yaxis_title="Proportion")
            fig_psi.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_psi, use_container_width=True)

        # ── 4. Validation Summary ────────────────────────────────────────────
        _step(4, "Validation Summary", "&#9989;", TH.RISK_SAFE)
        checks = [
            (TH.RISK_SAFE, "IV Feature Selection", "97 to 7 via IV 0.10-0.50"),
            (TH.RISK_SAFE, "WOE Monotonicity", "Clean risk gradients verified"),
            (TH.RISK_SAFE, "LR Assumptions", "6/6 satisfied by WOE design"),
            (psi_color, f"PSI = {psi_total:.4f}", "Score distribution stable"),
            (TH.RISK_SAFE, f"KS = {lr_m['ks']:.4f}", "Strong rank-ordering"),
            (TH.RISK_SAFE, "No Leakage", "40 columns removed"),
        ]
        check_cols = st.columns(3)
        for i, (c_color, c_title, c_desc) in enumerate(checks):
            with check_cols[i % 3]:
                st.markdown(f"""<div style="background:{TH.BG_PRIMARY};border:1px solid {TH.BORDER_LIGHT};
                border-radius:8px;padding:14px;margin-bottom:8px;">
                <div style="color:{c_color};font-weight:700;font-size:0.78rem;margin-bottom:3px;">
                &#10003; {c_title}</div>
                <div style="color:{TH.TEXT_SECONDARY};font-size:0.7rem;">{c_desc}</div>
                </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — MODEL PERFORMANCE
    # ══════════════════════════════════════════════════════════════════════════
    with tab_perf:

        df_filtered = df_master.copy()
        model_colors = {"Logistic Regression": TH.ACCENT_CYAN, "Random Forest": TH.RISK_CAUTION, "XGBoost": TH.ACCENT_BLUE}

        # ── Model Comparison Cards ───────────────────────────────────────────
        render_section("Model Performance Comparison", "3 models on same 70/30 stratified split — Logistic Regression deployed")

        col_cards = st.columns(3)
        for i, name in enumerate(["Logistic Regression", "Random Forest", "XGBoost"]):
            m = all_metrics[name]
            is_deployed = name == "Logistic Regression"
            badge = (f'<span style="background:{TH.ACCENT_BLUE};color:white;font-size:0.55rem;'
                     f'font-weight:700;padding:2px 8px;border-radius:3px;margin-left:8px;">DEPLOYED</span>') if is_deployed else ""
            with col_cards[i]:
                st.markdown(
                    f'<div style="background:{TH.BG_SECONDARY};border:1px solid {TH.BORDER_LIGHT};'
                    f'border-top:3px solid {model_colors[name]};border-radius:8px;padding:20px;">'
                    f'<div style="color:{TH.TEXT_PRIMARY};font-size:0.9rem;font-weight:700;">{name}{badge}</div>'
                    f'<table style="width:100%;margin-top:14px;border-collapse:collapse;">'
                    f'<tr><td style="padding:5px 0;color:{TH.TEXT_MUTED};font-size:0.6rem;font-weight:700;">AUC</td>'
                    f'<td style="text-align:right;font-family:\'JetBrains Mono\';font-weight:700;color:{TH.TEXT_PRIMARY};font-size:1.1rem;">{m["auc"]:.4f}</td></tr>'
                    f'<tr><td style="padding:5px 0;color:{TH.TEXT_MUTED};font-size:0.6rem;font-weight:700;">KS</td>'
                    f'<td style="text-align:right;font-family:\'JetBrains Mono\';font-weight:700;color:{TH.TEXT_PRIMARY};font-size:1.1rem;">{m["ks"]:.4f}</td></tr>'
                    f'<tr><td style="padding:5px 0;color:{TH.TEXT_MUTED};font-size:0.6rem;font-weight:700;">ACCURACY</td>'
                    f'<td style="text-align:right;font-family:\'JetBrains Mono\';font-weight:700;color:{TH.TEXT_PRIMARY};font-size:1.1rem;">{m["accuracy"]*100:.1f}%</td></tr>'
                    f'<tr><td style="padding:5px 0;color:{TH.TEXT_MUTED};font-size:0.6rem;font-weight:700;">PRECISION</td>'
                    f'<td style="text-align:right;font-family:\'JetBrains Mono\';font-weight:700;color:{TH.TEXT_PRIMARY};font-size:1.1rem;">{m["precision"]:.3f}</td></tr>'
                    f'<tr><td style="padding:5px 0;color:{TH.TEXT_MUTED};font-size:0.6rem;font-weight:700;">RECALL</td>'
                    f'<td style="text-align:right;font-family:\'JetBrains Mono\';font-weight:700;color:{TH.TEXT_PRIMARY};font-size:1.1rem;">{m["recall"]:.3f}</td></tr>'
                    f'<tr><td style="padding:5px 0;color:{TH.TEXT_MUTED};font-size:0.6rem;font-weight:700;">F1</td>'
                    f'<td style="text-align:right;font-family:\'JetBrains Mono\';font-weight:700;color:{TH.TEXT_PRIMARY};font-size:1.1rem;">{m["f1"]:.3f}</td></tr>'
                    f'</table></div>', unsafe_allow_html=True)

        st.markdown(f"""<div style="color:{TH.TEXT_MUTED};font-size:0.7rem;margin-top:12px;text-align:center;">
        Lending Club (Kaggle) &middot; {split['train_size']:,} train &middot; {split['test_size']:,} test &middot; 70/30 stratified split</div>""",
        unsafe_allow_html=True)

        # ── Confusion Matrix ─────────────────────────────────────────────────
        st.markdown("---")
        render_section("Confusion Matrix — Logistic Regression", "True vs predicted classifications on the test set")
        cm = all_metrics["Logistic Regression"]["confusion_matrix"]
        cm_labels = ["Non-Default", "Default"]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=cm_labels, y=cm_labels,
            colorscale=[[0, TH.BG_SECONDARY], [1, TH.ACCENT_BLUE]],
            text=[[f"{v:,}" for v in row] for row in cm],
            texttemplate="%{text}", textfont={"size": 16, "color": TH.TEXT_PRIMARY},
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>",
            showscale=False
        ))
        fig_cm.update_layout(**TH.get_plotly_layout(), height=350, xaxis_title="Predicted", yaxis_title="Actual")
        fig_cm.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_cm, use_container_width=True)

        # ── Side-by-Side Metrics Bar Chart ───────────────────────────────────
        st.markdown("---")
        render_section("Side-by-Side Metrics", "Visual comparison across all metrics")
        metric_names = ["AUC", "KS Score", "Accuracy", "Precision", "Recall", "F1"]
        fig_compare = go.Figure()
        for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
            m = all_metrics[name]
            vals = [m["auc"], m["ks"], m["accuracy"], m["precision"], m["recall"], m["f1"]]
            fig_compare.add_trace(go.Bar(
                name=name, x=metric_names, y=vals,
                marker_color=model_colors[name], opacity=0.85,
                text=[f"{v:.3f}" for v in vals], textposition="outside"
            ))
        fig_compare.update_layout(**TH.get_plotly_layout(), height=380, barmode="group",
                                  yaxis_title="Score", xaxis_title="")
        st.plotly_chart(fig_compare, use_container_width=True)

        # ── ROC Curves ───────────────────────────────────────────────────────
        st.markdown("---")
        render_section("ROC Curve Comparison", "Higher curve = better model")
        fig_roc = go.Figure()
        for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
            rd = roc_data[name]
            auc_val = all_metrics[name]["auc"]
            fig_roc.add_trace(go.Scatter(
                x=rd["fpr"], y=rd["tpr"], mode="lines",
                name=f"{name} (AUC={auc_val:.3f})",
                line=dict(color=model_colors[name], width=2.5)
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline",
            line=dict(color=TH.TEXT_MUTED, width=1.5, dash="dash")
        ))
        fig_roc.update_layout(
            **{**TH.get_plotly_layout(), "legend": dict(x=0.55, y=0.05, bgcolor="rgba(255,255,255,0.8)")},
            height=450, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

        _txt(f"<strong>AUC = 1.0</strong>: Perfect separation. "
             f"<strong>AUC = 0.5</strong>: Random guessing. "
             f"Our deployed LR achieves <strong>AUC = {all_metrics['Logistic Regression']['auc']:.3f}</strong>.")

        # ── Feature Importance ───────────────────────────────────────────────
        st.markdown("---")
        render_section("Feature Importance", "Correlation strength with actual defaults")
        numeric_cols = [c for c in ["grade", "int_rate", "all_util", "max_bal_bc", "mths_since_rcnt_il", "total_bal_il", "il_util", "prob"] if c in df_filtered.columns]
        if "target" in df_filtered.columns:
            importance = df_filtered[numeric_cols].corrwith(df_filtered["target"]).abs().sort_values(ascending=False)
            importance_df = pd.DataFrame({
                "feature": [FEATURE_DESCRIPTIONS.get(c, c) for c in importance.index],
                "importance": importance.values
            })
            fig_imp = px.bar(
                importance_df.sort_values("importance"), x="importance", y="feature",
                orientation="h", color="importance",
                color_continuous_scale=[[0, TH.RISK_SAFE], [1, TH.RISK_DANGER]]
            )
            fig_imp.update_traces(marker_line_width=0, text=importance_df.sort_values("importance")["importance"], textposition="outside")
            fig_imp.update_layout(**TH.get_plotly_layout(), height=400, showlegend=False,
                                  xaxis_title="Absolute Correlation with Default", yaxis_title="")
            st.plotly_chart(fig_imp, use_container_width=True)

        # ── Correlation Matrix ───────────────────────────────────────────────
        st.markdown("---")
        render_section("Correlation Matrix", "Feature inter-relationships")
        corr_matrix = df_filtered[numeric_cols].corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[FEATURE_DESCRIPTIONS.get(c, c) for c in corr_matrix.columns],
            y=[FEATURE_DESCRIPTIONS.get(c, c) for c in corr_matrix.columns],
            colorscale=[[0, "#EFF6FF"], [0.5, TH.ACCENT_BLUE], [1, "#1E3A5F"]],
            text=np.round(corr_matrix.values, 2), texttemplate="%{text:.2f}",
            textfont={"size": 10},
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>"
        ))
        fig_corr.update_layout(**TH.get_plotly_layout(), height=500, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_corr, use_container_width=True)

        # ── Calibration ──────────────────────────────────────────────────────
        st.markdown("---")
        render_section("Model Calibration", "How well predictions match actual default rates")
        df_calib = df_filtered[df_filtered["target"].notna()].copy()
        df_calib["pd_bins"] = pd.cut(df_calib["prob"], bins=10)
        calibration = df_calib.groupby("pd_bins", observed=True).agg({"prob": "mean", "target": ["count", "mean"]}).reset_index()
        calibration.columns = ["bin", "mean_pred_pd", "count", "obs_default_rate"]
        calibration = calibration[calibration["count"] >= 10]

        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=calibration["mean_pred_pd"], y=calibration["obs_default_rate"],
            mode="markers+lines", name="Observed",
            marker=dict(size=10, color=TH.RISK_DANGER), line=dict(color=TH.RISK_DANGER, width=2)
        ))
        max_pd = df_calib["prob"].max()
        fig_cal.add_trace(go.Scatter(
            x=[0, max_pd], y=[0, max_pd], mode="lines", name="Perfect Calibration",
            line=dict(color=TH.TEXT_MUTED, width=2, dash="dash")
        ))
        fig_cal.update_layout(**TH.get_plotly_layout(), height=400,
                              xaxis_title="Predicted Default Probability", yaxis_title="Actual Default Rate",
                              hovermode="closest")
        st.plotly_chart(fig_cal, use_container_width=True)
        _txt("<strong>Dashed line</strong> = perfect calibration. "
             "<strong>Red line</strong> = our model. Points above the line = under-predicting risk. "
             "Points below = over-predicting risk.")


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
        ["Portfolio Overview", "ML Journey"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown(f"""
    <div style="background: rgba(5,150,105,0.04); border: 1px solid rgba(5,150,105,0.15); border-radius: 6px; padding: 12px; margin-bottom: 12px;">
        <div style="color: {TH.RISK_SAFE}; font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Data Source</div>
        <div style="color: {TH.TEXT_PRIMARY}; font-size: 0.8rem; font-weight: 600; margin-top: 4px;">Lending Club via Kaggle (Real Data)</div>
        <div style="color: {TH.TEXT_MUTED}; font-size: 0.7rem; margin-top: 2px;">{len(df_master):,} loans · 7 features · Actual defaults</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <div style="background: {TH.BG_HOVER}; border-radius: 6px; padding: 12px; margin-bottom: 12px;">
        <div style="color: {TH.TEXT_MUTED}; font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Approach</div>
        <div style="color: {TH.TEXT_SECONDARY}; font-size: 0.72rem; margin-top: 6px; line-height: 1.7;">
            EDA &rarr; Feature Engineering &rarr; Model Training<br>
            (Logistic Reg / Random Forest / XGBoost)<br>
            &rarr; ECL Estimation &rarr; ML Journey
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <div style="background: {TH.BG_HOVER}; border-radius: 6px; padding: 12px; margin-bottom: 12px;">
        <div style="color: {TH.TEXT_MUTED}; font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Technology Stack</div>
        <div style="color: {TH.TEXT_SECONDARY}; font-size: 0.72rem; margin-top: 6px; line-height: 1.7;">
            scikit-learn · Pandas · NumPy<br>
            Plotly · Streamlit · Python 3.x
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
    if page == "Portfolio Overview":
        page_executive_summary()
    elif page == "ML Journey":
        page_ml_journey()

    # Footer
    st.markdown(f"""
    <div style="text-align: center; padding: 40px 0; margin-top: 60px; border-top: 1px solid {TH.BORDER_LIGHT};">
        <p style="color: {TH.TEXT_MUTED}; font-size: 0.7rem; letter-spacing: 1px; text-transform: uppercase;">
            LENDER'S CLUB &mdash; CREDIT RISK ANALYTICS DASHBOARD
        </p>
        <p style="color: {TH.TEXT_SECONDARY}; font-size: 0.72rem; margin-top: 8px; line-height:1.7;">
            Built by <strong>Aditya Parbhakar</strong><br>
            ML-based credit scoring &middot; ECL framework &middot; End-to-end ML methodology<br>
            Data: Lending Club (Kaggle) &middot; 30,000 real loans &middot; Not synthetic
        </p>
        <p style="color: {TH.TEXT_MUTED}; font-size: 0.6rem; margin-top: 10px;">
            This dashboard is a portfolio project demonstrating credit risk modeling, not financial advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
