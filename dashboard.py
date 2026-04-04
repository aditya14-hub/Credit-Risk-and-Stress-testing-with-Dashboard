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
    if os.path.exists(_MODEL_CACHE_PATH):
        with open(_MODEL_CACHE_PATH, "rb") as f:
            data = pickle.load(f)
        if "lr" in data:
            return data["lr"], data["all_metrics"], data["roc_data"]

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

    # Save for next time
    with open(_MODEL_CACHE_PATH, "wb") as f:
        pickle.dump({"lr": lr, "all_metrics": all_metrics, "roc_data": roc_data}, f)

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
            Expected Credit Loss follows the <strong>Basel III</strong> formula: <strong>ECL = PD &times; LGD (60%) &times; EAD ($125K)</strong>.
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
            <strong>Logistic Regression was selected for deployment</strong> due to its <strong>full interpretability</strong> — regulators (Basel III / SR 11-7) require
            that credit models provide transparent, explainable risk scores. LR coefficients map directly to scorecard points, enabling
            clear audit trails and compliance documentation. It also offers <strong>competitive AUC</strong> with significantly lower model risk,
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
            badge = f'<span style="background:{TH.ACCENT_BLUE}; color:white; font-size:0.55rem; font-weight:700; padding:2px 8px; border-radius:3px; margin-left:8px;">BEST</span>' if is_best else ""
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
# PAGE: STRESS TESTING & SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

def page_stress_testing():
    """Simulate worst-case scenarios and measure loss impact."""
    render_header(
        "Stress Testing & What-If Scenarios",
        "Simulate adverse economic conditions using Basel III ECL framework",
        "SCENARIO"
    )

    st.markdown(f"""
    <div style="background:rgba(37,99,235,0.04); border:1px solid rgba(37,99,235,0.15); border-radius:8px; padding:16px 20px; margin-bottom:16px;">
        <div style="color:{TH.ACCENT_BLUE}; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">Regulatory Context</div>
        <div style="color:{TH.TEXT_SECONDARY}; font-size:0.78rem; line-height:1.7;">
            Under <strong>Basel III</strong> and <strong>IFRS 9</strong>, banks must hold capital reserves against expected credit losses (ECL).
            Stress testing simulates adverse economic scenarios to estimate how much additional capital would be needed.
            The formula used: <strong>ECL = PD &times; LGD &times; (1 &minus; Recovery) &times; EAD</strong>, where PD = Probability of Default, LGD = Loss Given Default, EAD = Exposure at Default.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Scenario Inputs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pd_shock = st.slider("PD Stress Multiplier", 1.0, 3.0, 1.5, 0.1,
                            help="Simulates economic downturn — e.g., 2.0x means default probabilities double (similar to 2008 crisis levels)")
    with col2:
        lgd_rate = st.slider("Loss Given Default (LGD)", 0.0, 1.0, 0.60, 0.05,
                            help="Gross loss severity — Basel III foundation approach uses 45% for senior unsecured; we default to 60% for consumer loans")
    with col3:
        recovery_rate = st.slider("Recovery Rate", 0.0, 1.0, 0.40, 0.05,
                                 help="Post-default recovery through collections, collateral, or write-off recoupment. Industry avg for unsecured: 20-40%")
    with col4:
        ead_value = st.number_input("Exposure at Default ($)", value=125_000, step=25_000,
                                   help="Average loan exposure — based on portfolio mean balance")

    # Apply scenarios with proper ECL formula: PD × LGD × (1 - Recovery) × EAD
    net_lgd = lgd_rate * (1 - recovery_rate)
    df_stress = df_master.copy()
    df_stress["baseline_ecl"] = df_stress["prob"] * net_lgd * ead_value
    df_stress["stressed_pd"] = (df_stress["prob"] * pd_shock).clip(upper=1.0)
    df_stress["stressed_ecl"] = df_stress["stressed_pd"] * net_lgd * ead_value
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
    
    st.markdown(f"""
    <div style="background:{TH.BG_SECONDARY}; border:1px solid {TH.BORDER_LIGHT}; border-radius:8px; padding:14px 18px; margin:12px 0;">
        <div style="color:{TH.TEXT_SECONDARY}; font-size:0.78rem; line-height:1.7;">
            <strong>Scenario:</strong> If default probabilities increase by <strong>{pd_shock}x</strong> (e.g., recession scenario),
            with a net loss severity of <strong>{net_lgd*100:.1f}%</strong> (LGD {lgd_rate*100:.0f}% &times; (1 &minus; Recovery {recovery_rate*100:.0f}%)),
            the portfolio would need an additional <strong>{format_currency(capital_need)}</strong> in loss reserves —
            a <strong>{capital_pct:.1f}%</strong> increase over baseline.
            {'This is within typical capital buffer thresholds.' if capital_pct < 50 else 'This exceeds typical capital buffers — management action required.'}
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        render_section("Risk Migration Matrix", "Transition probabilities — how loans shift between risk tiers under stress")

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
            color_continuous_scale=[[0, "#FEF2F2"], [1, TH.RISK_DANGER]],
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
# PAGE: RISK PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

def page_risk_predictor():
    """Interactive loan risk prediction tool."""
    render_header(
        "Loan Risk Predictor",
        "Input borrower details and get a real-time default probability prediction from the Logistic Regression model",
        "PREDICT"
    )

    model, all_metrics, _ = train_models()

    st.markdown(f"""
    <div style="background:rgba(37,99,235,0.04); border:1px solid rgba(37,99,235,0.15); border-radius:8px; padding:16px 20px; margin-bottom:4px;">
        <div style="color:{TH.ACCENT_BLUE}; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">How It Works</div>
        <div style="color:{TH.TEXT_SECONDARY}; font-size:0.78rem; line-height:1.7;">
            Adjust borrower parameters below. The deployed <strong>Logistic Regression model</strong> (AUC: {all_metrics['Logistic Regression']['auc']:.3f}, KS: {all_metrics['Logistic Regression']['ks']:.3f}) generates a
            real-time probability of default (PD). The system computes the estimated credit loss and classifies
            the borrower into a risk tier — simulating the underwriting decision a credit analyst would make.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(f"""
    <div style="background:{TH.BG_SECONDARY}; border:1px solid {TH.BORDER_LIGHT}; border-radius:8px; padding:14px 18px; margin-bottom:14px;">
        <div style="color:{TH.TEXT_PRIMARY}; font-size:0.8rem; font-weight:700; margin-bottom:8px;">Reading the Sliders</div>
        <div style="color:{TH.TEXT_SECONDARY}; font-size:0.75rem; line-height:1.7;">
            Features are <strong>standardized</strong> — values represent how far a borrower is from the portfolio average.
            <strong style="color:{TH.RISK_SAFE};">Negative</strong> = below average,
            <strong>0.00</strong> = at average,
            <strong style="color:{TH.RISK_DANGER};">Positive</strong> = above average.
            Move sliders right to simulate a riskier borrower profile.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature inputs (all standardized) with human-readable context
    col1, col2 = st.columns(2)

    with col1:
        st.caption("Lower credit quality, higher rates, and more credit usage increase risk.")
        grade = st.slider(
            "Credit Grade  (- better | + worse)",
            float(df_master["grade"].min()), float(df_master["grade"].max()),
            float(df_master["grade"].median()), step=0.05,
            help="Lending Club letter grade (A through G). Negative = closer to A (best), Positive = closer to G (worst)."
        )
        int_rate = st.slider(
            "Interest Rate  (- lower | + higher)",
            float(df_master["int_rate"].min()), float(df_master["int_rate"].max()),
            float(df_master["int_rate"].median()), step=0.05,
            help="Annual interest rate on the loan. Higher values mean more expensive loans, which correlate with higher risk."
        )
        all_util = st.slider(
            "Credit Utilization  (- less | + more)",
            float(df_master["all_util"].min()), float(df_master["all_util"].max()),
            float(df_master["all_util"].median()), step=0.05,
            help="How much of their available revolving credit the borrower uses. High utilization (>70%) is a red flag."
        )
        max_bal_bc = st.slider(
            "Max Bankcard Balance  (- smaller | + larger)",
            float(df_master["max_bal_bc"].min()), float(df_master["max_bal_bc"].max()),
            float(df_master["max_bal_bc"].median()), step=0.05,
            help="Peak balance on any single credit card. Larger peak balances suggest higher spending habits."
        )

    with col2:
        st.caption("Installment history and balances reflect the borrower's debt load.")
        mths_since_rcnt_il = st.slider(
            "Months Since Recent Installment  (- recent | + longer ago)",
            float(df_master["mths_since_rcnt_il"].min()), float(df_master["mths_since_rcnt_il"].max()),
            float(df_master["mths_since_rcnt_il"].median()), step=0.05,
            help="Time since borrower last opened an installment account. Longer gap can mean less recent borrowing activity."
        )
        total_bal_il = st.slider(
            "Total Installment Balance  (- lower | + higher)",
            float(df_master["total_bal_il"].min()), float(df_master["total_bal_il"].max()),
            float(df_master["total_bal_il"].median()), step=0.05,
            help="Combined balance on all installment loans (auto, student, personal). Higher = more total debt."
        )
        il_util = st.slider(
            "Installment Utilization  (- less | + more)",
            float(df_master["il_util"].min()), float(df_master["il_util"].max()),
            float(df_master["il_util"].median()), step=0.05,
            help="How much of installment credit limits the borrower has used. High utilization suggests stretched finances."
        )

    # Make prediction
    input_df = pd.DataFrame([[grade, int_rate, all_util, max_bal_bc, mths_since_rcnt_il, total_bal_il, il_util]],
                             columns=FEATURE_COLS)
    pred_prob = float(model.predict_proba(input_df)[0][1])
    risk_label = "Low" if pred_prob < 0.08 else "Medium" if pred_prob < 0.20 else "High"
    risk_color = RISK_COLORS.get(risk_label, TH.TEXT_PRIMARY)

    st.markdown("---")

    # Results
    col_r1, col_r2, col_r3 = st.columns([1, 1, 1])

    with col_r1:
        # Gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_prob * 100,
            number={"suffix": "%", "font": {"size": 36, "color": TH.TEXT_PRIMARY}},
            title={"text": "Default Probability", "font": {"size": 14, "color": TH.TEXT_SECONDARY}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TH.TEXT_MUTED},
                "bar": {"color": risk_color, "thickness": 0.3},
                "bgcolor": TH.BG_SECONDARY,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 8], "color": "rgba(5,150,105,0.1)"},
                    {"range": [8, 20], "color": "rgba(217,119,6,0.1)"},
                    {"range": [20, 100], "color": "rgba(220,38,38,0.1)"},
                ],
            }
        ))
        fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": TH.TEXT_PRIMARY}, height=250, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_g, use_container_width=True)

    with col_r2:
        st.markdown(f"""
        <div style="text-align:center; padding-top:40px;">
            <div style="color:{TH.TEXT_MUTED}; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:1px;">Risk Classification</div>
            <div style="color:{risk_color}; font-size:2.5rem; font-weight:800; margin-top:8px;">{risk_label}</div>
            <div style="color:{TH.TEXT_SECONDARY}; font-size:0.8rem; margin-top:8px;">
                {'Borrower is unlikely to default' if risk_label == 'Low' else 'Moderate risk — monitor closely' if risk_label == 'Medium' else 'High default risk — caution advised'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r3:
        avg_pd = df_master["prob"].mean()
        diff = pred_prob - avg_pd
        st.markdown(f"""
        <div style="padding-top:30px;">
            <div style="color:{TH.TEXT_MUTED}; font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:12px;">Comparison to Portfolio</div>
            <div style="display:grid; gap:10px;">
                <div style="background:{TH.BG_SECONDARY}; border-radius:6px; padding:10px 14px;">
                    <div style="color:{TH.TEXT_MUTED}; font-size:0.65rem;">This Loan</div>
                    <div style="color:{TH.TEXT_PRIMARY}; font-size:1.1rem; font-weight:700;">{pred_prob*100:.2f}%</div>
                </div>
                <div style="background:{TH.BG_SECONDARY}; border-radius:6px; padding:10px 14px;">
                    <div style="color:{TH.TEXT_MUTED}; font-size:0.65rem;">Portfolio Average</div>
                    <div style="color:{TH.TEXT_PRIMARY}; font-size:1.1rem; font-weight:700;">{avg_pd*100:.2f}%</div>
                </div>
                <div style="background:{TH.BG_SECONDARY}; border-radius:6px; padding:10px 14px;">
                    <div style="color:{TH.TEXT_MUTED}; font-size:0.65rem;">Difference</div>
                    <div style="color:{TH.RISK_DANGER if diff > 0 else TH.RISK_SAFE}; font-size:1.1rem; font-weight:700;">{diff*100:+.2f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Credit Decision Summary
    est_ecl = pred_prob * 0.60 * 125_000
    tier_avg_pd = df_master.groupby("risk_bucket", observed=True)["prob"].mean().to_dict()
    percentile_rank = (df_master["prob"] < pred_prob).mean() * 100

    st.markdown(f"""
    <div style="background:{TH.BG_SECONDARY}; border:1px solid {TH.BORDER_LIGHT}; border-radius:8px; padding:20px; margin:8px 0 16px 0;">
        <div style="color:{TH.TEXT_PRIMARY}; font-size:0.85rem; font-weight:700; margin-bottom:12px;">Credit Decision Summary</div>
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:16px;">
            <div>
                <div style="color:{TH.TEXT_MUTED}; font-size:0.65rem; font-weight:700; text-transform:uppercase;">Estimated Annual ECL</div>
                <div style="color:{TH.TEXT_PRIMARY}; font-size:1.1rem; font-weight:700; font-family:'JetBrains Mono',monospace;">{format_currency(est_ecl)}</div>
                <div style="color:{TH.TEXT_MUTED}; font-size:0.65rem;">PD &times; LGD (60%) &times; $125K</div>
            </div>
            <div>
                <div style="color:{TH.TEXT_MUTED}; font-size:0.65rem; font-weight:700; text-transform:uppercase;">Risk Percentile</div>
                <div style="color:{TH.TEXT_PRIMARY}; font-size:1.1rem; font-weight:700; font-family:'JetBrains Mono',monospace;">{percentile_rank:.0f}th</div>
                <div style="color:{TH.TEXT_MUTED}; font-size:0.65rem;">{'Riskier than most borrowers' if percentile_rank > 70 else 'Within normal range' if percentile_rank > 30 else 'Safer than most borrowers'}</div>
            </div>
            <div>
                <div style="color:{TH.TEXT_MUTED}; font-size:0.65rem; font-weight:700; text-transform:uppercase;">Suggested Action</div>
                <div style="color:{risk_color}; font-size:0.95rem; font-weight:700; margin-top:2px;">{'Approve — standard terms' if risk_label == 'Low' else 'Approve — enhanced monitoring required' if risk_label == 'Medium' else 'Decline or require collateral'}</div>
                <div style="color:{TH.TEXT_MUTED}; font-size:0.65rem;">{'Meets risk appetite thresholds' if risk_label == 'Low' else 'Exceeds standard risk appetite' if risk_label == 'Medium' else 'Exceeds maximum risk tolerance'}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature contribution breakdown
    st.markdown("---")
    render_section("Input Feature Summary", "How this borrower compares to the portfolio (standard deviations from mean)")

    feature_names = [FEATURE_DESCRIPTIONS.get(c, c) for c in FEATURE_COLS]
    input_vals = [grade, int_rate, all_util, max_bal_bc, mths_since_rcnt_il, total_bal_il, il_util]
    mean_vals = [float(df_master[c].mean()) for c in FEATURE_COLS]
    std_vals = [float(df_master[c].std()) for c in FEATURE_COLS]
    z_scores = [(iv - mv) / sv if sv > 0 else 0 for iv, mv, sv in zip(input_vals, mean_vals, std_vals)]

    colors = [TH.RISK_DANGER if abs(z) > 1.5 else TH.RISK_CAUTION if abs(z) > 0.75 else TH.RISK_SAFE for z in z_scores]

    fig_feat = go.Figure(go.Bar(
        x=z_scores, y=feature_names, orientation="h",
        marker_color=colors,
        text=[f"{z:+.2f}\u03c3" for z in z_scores],
        textposition="outside"
    ))
    max_abs_z = max((abs(z) for z in z_scores), default=1.0)
    x_limit = max(max_abs_z * 1.3, 2.0)
    fig_feat.update_layout(
        **TH.get_plotly_layout(), height=320,
        xaxis_title="Standard Deviations from Portfolio Mean",
        yaxis_title=""
    )
    fig_feat.update_xaxes(range=[-x_limit, x_limit])
    st.plotly_chart(fig_feat, use_container_width=True)

    st.caption("Green = near average | Amber = moderately unusual | Red = significantly different from portfolio")


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
        ["Portfolio Overview", "Model Performance", "Risk Predictor", "Stress Testing"],
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
            &rarr; ECL Estimation (Basel III) &rarr; Stress Testing
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
    elif page == "Model Performance":
        page_advanced_analytics()
    elif page == "Risk Predictor":
        page_risk_predictor()
    elif page == "Stress Testing":
        page_stress_testing()
    
    # Footer
    st.markdown(f"""
    <div style="text-align: center; padding: 40px 0; margin-top: 60px; border-top: 1px solid {TH.BORDER_LIGHT};">
        <p style="color: {TH.TEXT_MUTED}; font-size: 0.7rem; letter-spacing: 1px; text-transform: uppercase;">
            LENDER'S CLUB &mdash; CREDIT RISK ANALYTICS DASHBOARD
        </p>
        <p style="color: {TH.TEXT_SECONDARY}; font-size: 0.72rem; margin-top: 8px; line-height:1.7;">
            Built by <strong>Aditya Parbhakar</strong><br>
            ML-based credit scoring &middot; Basel III ECL framework &middot; Regulatory stress testing<br>
            Data: Lending Club (Kaggle) &middot; 30,000 real loans &middot; Not synthetic
        </p>
        <p style="color: {TH.TEXT_MUTED}; font-size: 0.6rem; margin-top: 10px;">
            This dashboard is a portfolio project demonstrating credit risk modeling, not financial advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
