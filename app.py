import streamlit as st
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import io

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Employee Turnover Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# ULTRA DECORATIVE BACKGROUND LAYER (HTML)
# =============================================================================
st.markdown(
    """
<div class="ultra-bg" aria-hidden="true">
  <div class="blob b1"></div>
  <div class="blob b2"></div>
  <div class="blob b3"></div>
  <div class="grain"></div>
  <div class="stars"></div>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# JAW-DROPPING CSS (GLASSMORPHISM + NEON + ANIMATED BACKGROUND)
# =============================================================================
st.markdown(
    r"""
<style>
/* -------------------------
   Fonts
-------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Poppins:wght@300;400;600;700;800;900&display=swap');

/* -------------------------
   Global App Background
-------------------------- */
html, body {
    height: 100%;
}

div[data-testid="stAppViewContainer"] {
    background: radial-gradient(1200px 700px at 10% 10%, rgba(255, 0, 140, 0.16), transparent 55%),
                radial-gradient(900px 600px at 90% 20%, rgba(64, 224, 208, 0.18), transparent 55%),
                radial-gradient(900px 700px at 50% 90%, rgba(255, 200, 0, 0.15), transparent 60%),
                linear-gradient(135deg, #070A12 0%, #0A1024 35%, #070A12 100%) !important;
    font-family: "Poppins", system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
}

div[data-testid="stHeader"] {
    background: transparent !important;
}

div[data-testid="stToolbar"] {
    right: 1rem;
}

/* Center and widen content */
.block-container {
    max-width: 1250px !important;
    padding-left: 4.2rem !important;
    padding-right: 4.2rem !important;
    padding-top: 2.2rem !important;
    margin: 0 auto !important;
}

/* -------------------------
   Ultra Background Layer
-------------------------- */
.ultra-bg {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
}

.blob {
    position: absolute;
    width: 720px;
    height: 720px;
    filter: blur(38px);
    opacity: 0.6;
    transform: translateZ(0);
    border-radius: 55% 45% 60% 40% / 55% 40% 60% 45%;
    animation: blobFloat 10s ease-in-out infinite;
}
.b1 {
    left: -180px;
    top: -220px;
    background: radial-gradient(circle at 30% 30%, rgba(255, 0, 128, 0.95), rgba(255, 140, 0, 0.35), transparent 65%);
}
.b2 {
    right: -220px;
    top: -160px;
    background: radial-gradient(circle at 40% 40%, rgba(0, 245, 255, 0.85), rgba(255, 0, 255, 0.35), transparent 65%);
    animation-delay: -3.5s;
}
.b3 {
    left: 20%;
    bottom: -420px;
    background: radial-gradient(circle at 40% 40%, rgba(255, 215, 0, 0.75), rgba(64, 224, 208, 0.25), transparent 70%);
    animation-delay: -6s;
}

@keyframes blobFloat {
    0%   { transform: translate(0px, 0px) scale(1) rotate(0deg); }
    33%  { transform: translate(35px, 25px) scale(1.03) rotate(7deg); }
    66%  { transform: translate(-25px, 35px) scale(0.98) rotate(-6deg); }
    100% { transform: translate(0px, 0px) scale(1) rotate(0deg); }
}

/* subtle grain overlay */
.grain {
    position: absolute;
    inset: -50%;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='240' height='240'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='240' height='240' filter='url(%23n)' opacity='.25'/%3E%3C/svg%3E");
    opacity: 0.06;
    animation: grainMove 10s steps(2) infinite;
    mix-blend-mode: overlay;
}
@keyframes grainMove {
    0% { transform: translate(0,0); }
    25% { transform: translate(-2%, 3%); }
    50% { transform: translate(-5%, -2%); }
    75% { transform: translate(3%, -5%); }
    100% { transform: translate(0,0); }
}

/* star sparkles */
.stars {
    position: absolute;
    inset: 0;
    background:
        radial-gradient(1px 1px at 10% 20%, rgba(255,255,255,0.65), transparent 40%),
        radial-gradient(1px 1px at 70% 30%, rgba(255,255,255,0.55), transparent 40%),
        radial-gradient(1px 1px at 40% 80%, rgba(255,255,255,0.55), transparent 40%),
        radial-gradient(1px 1px at 85% 70%, rgba(255,255,255,0.60), transparent 40%),
        radial-gradient(1px 1px at 25% 60%, rgba(255,255,255,0.50), transparent 40%);
    opacity: 0.6;
    animation: twinkle 2.8s ease-in-out infinite;
}
@keyframes twinkle {
    0%, 100% { opacity: 0.55; }
    50%      { opacity: 0.9; }
}

/* Ensure app content stays above background */
main, header, section, .block-container {
    position: relative;
    z-index: 1;
}

/* -------------------------
   Hero Header
-------------------------- */
.hero-wrap {
    position: relative;
    border-radius: 22px;
    padding: 2.0rem 2.0rem;
    margin-bottom: 1.6rem;
    background: linear-gradient(135deg,
        rgba(255, 255, 255, 0.10) 0%,
        rgba(255, 255, 255, 0.06) 50%,
        rgba(255, 255, 255, 0.08) 100%);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow:
        0 18px 60px rgba(0,0,0,0.35),
        0 0 0 1px rgba(255,255,255,0.04) inset;
    overflow: hidden;
}

.hero-wrap::before {
    content: "";
    position: absolute;
    inset: -2px;
    background: conic-gradient(from 220deg,
        #ff00cc, #3333ff, #00f5ff, #ffd000, #ff0080, #ff00cc);
    filter: blur(18px);
    opacity: 0.28;
    animation: heroGlow 4s linear infinite;
}
@keyframes heroGlow {
    0% { transform: rotate(0deg) scale(1); }
    50% { transform: rotate(180deg) scale(1.02); }
    100% { transform: rotate(360deg) scale(1); }
}

.hero {
    position: relative;
    display: grid;
    grid-template-columns: 1.2fr 0.8fr;
    gap: 1rem;
    align-items: center;
}

.hero-title {
    font-family: "Space Grotesk", "Poppins", sans-serif;
    font-size: 3.0rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    line-height: 1.05;
    margin: 0;
    color: rgba(255,255,255,0.92);
    text-shadow: 0 12px 42px rgba(0,0,0,0.45);
}
.hero-title span {
    background: linear-gradient(90deg, #00f5ff, #ff00ff, #ffd000, #00f5ff);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.hero-sub {
    margin-top: 0.7rem;
    font-size: 1.05rem;
    color: rgba(255,255,255,0.72);
    max-width: 56ch;
}

.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 1rem;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.55rem 0.85rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.88rem;
    color: rgba(255,255,255,0.9);
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

.hero-right {
    border-radius: 18px;
    padding: 1.1rem 1.1rem;
    background: linear-gradient(135deg, rgba(0,245,255,0.12), rgba(255,0,255,0.10));
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 14px 40px rgba(0,0,0,0.32);
}
.kpi {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
}
.kpi-card {
    border-radius: 14px;
    padding: 0.9rem 0.9rem;
    background: rgba(10,16,36,0.55);
    border: 1px solid rgba(255,255,255,0.10);
}
.kpi-top {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.65);
    font-weight: 700;
    letter-spacing: 0.3px;
}
.kpi-val {
    margin-top: 0.3rem;
    font-size: 1.5rem;
    font-weight: 900;
    color: rgba(255,255,255,0.92);
}
.kpi-foot {
    margin-top: 0.2rem;
    font-size: 0.8rem;
    color: rgba(255,255,255,0.6);
}

/* -------------------------
   Section Headers
-------------------------- */
.section-header {
    font-family: "Space Grotesk", "Poppins", sans-serif;
    font-size: 1.65rem;
    font-weight: 900;
    letter-spacing: 0.4px;
    color: rgba(255,255,255,0.92);
    text-align: center;
    margin: 1rem 0 1.2rem 0;
    text-shadow: 0 10px 35px rgba(0,0,0,0.35);
}

/* -------------------------
   Tab Styling (Neon Pills)
-------------------------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 16px;
    justify-content: center;
    background-color: transparent;
    padding: 0.8rem 0;
}

.stTabs [data-baseweb="tab"] {
    height: 92px !important;
    min-width: 360px !important;
    padding: 0 48px !important;
    border-radius: 18px !important;
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    font-weight: 900 !important;
    font-size: 1.35rem !important;
    color: rgba(255,255,255,0.86) !important;
    box-shadow: 0 14px 40px rgba(0,0,0,0.22);
    transition: transform .25s ease, box-shadow .25s ease, background .25s ease !important;
    position: relative;
    overflow: hidden;
}

.stTabs [data-baseweb="tab"]::before {
    content: "";
    position: absolute;
    inset: -2px;
    background: linear-gradient(90deg, #00f5ff, #ff00ff, #ffd000, #00f5ff);
    opacity: 0.0;
    filter: blur(10px);
    transition: opacity .25s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    transform: translateY(-3px) scale(1.01);
    box-shadow: 0 18px 55px rgba(0,0,0,0.32);
    background: rgba(255,255,255,0.085) !important;
}
.stTabs [data-baseweb="tab"]:hover::before { opacity: 0.22; }

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,245,255,0.22), rgba(255,0,255,0.18)) !important;
    color: rgba(255,255,255,0.96) !important;
    border: 1px solid rgba(0,245,255,0.35) !important;
    box-shadow:
        0 22px 70px rgba(0,0,0,0.40),
        0 0 0 1px rgba(255,255,255,0.08) inset,
        0 0 35px rgba(0,245,255,0.18);
}
.stTabs [aria-selected="true"]::before { opacity: 0.30; }

.stTabs [data-baseweb="tab-panel"] { padding-top: 16px; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* -------------------------
   Input Feature Cards (3D Glass)
-------------------------- */
.feature-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.11), rgba(255,255,255,0.06));
    border-radius: 16px;
    padding: 1.05rem 1.15rem;
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 18px 55px rgba(0,0,0,0.28);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    margin-bottom: 0.6rem;
    position: relative;
    overflow: hidden;
}
.feature-card::after {
    content: "";
    position: absolute;
    top: -40%;
    left: -30%;
    width: 80%;
    height: 180%;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,0.18), transparent);
    transform: rotate(18deg);
    animation: sheen 3.8s ease-in-out infinite;
    opacity: 0.55;
}
@keyframes sheen {
    0% { transform: translateX(-35%) rotate(18deg); opacity: 0.0; }
    20% { opacity: 0.55; }
    50% { transform: translateX(85%) rotate(18deg); opacity: 0.35; }
    100% { transform: translateX(130%) rotate(18deg); opacity: 0.0; }
}

/* Make Streamlit widgets match dark glass */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] div,
div[data-testid="stSlider"] {
    color: rgba(255,255,255,0.92) !important;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    border-radius: 12px !important;
}

div[data-testid="stSelectbox"] > div {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
}

div[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.06) !important;
    border: 1px dashed rgba(255,255,255,0.22) !important;
    border-radius: 16px !important;
}

/* -------------------------
   Prediction Result Boxes (Neon Glass)
-------------------------- */
.prediction-box {
    padding: 1.6rem 1.6rem;
    border-radius: 18px;
    text-align: center;
    margin: 0.6rem 0;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 22px 70px rgba(0,0,0,0.35);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    position: relative;
    overflow: hidden;
}
.prediction-box::before {
    content: "";
    position: absolute;
    inset: -2px;
    background: conic-gradient(from 180deg, rgba(0,245,255,0.0), rgba(0,245,255,0.55), rgba(255,0,255,0.55), rgba(255,208,0,0.55), rgba(0,245,255,0.0));
    filter: blur(18px);
    opacity: 0.22;
    animation: spin 4.2s linear infinite;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.stay-prediction { border-color: rgba(40, 167, 69, 0.55) !important; }
.leave-prediction { border-color: rgba(220, 53, 69, 0.55) !important; }

/* -------------------------
   Progress Bars (Neon)
-------------------------- */
.progress-bar-container {
    width: 100%;
    background: rgba(255,255,255,0.10);
    border-radius: 999px;
    margin: 8px 0 14px 0;
    height: 16px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.14);
}
.progress-bar-green {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(40,167,69,0.85), rgba(0,245,255,0.65));
    box-shadow: 0 0 18px rgba(40,167,69,0.35);
    transition: width 0.5s ease-in-out;
}
.progress-bar-red {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(220,53,69,0.85), rgba(255,0,255,0.60));
    box-shadow: 0 0 18px rgba(220,53,69,0.35);
    transition: width 0.5s ease-in-out;
}

/* -------------------------
   Stats Cards (Glass + Glow)
-------------------------- */
.stats-card {
    background: rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.2rem 1.1rem;
    box-shadow: 0 18px 60px rgba(0,0,0,0.30);
    text-align: center;
    border: 1px solid rgba(255,255,255,0.14);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
}
.stats-card h4 {
    color: rgba(255,255,255,0.80);
    margin: 0 0 0.45rem 0;
    font-weight: 800;
}
.stats-card .number {
    font-size: 2.1rem;
    font-weight: 900;
    color: rgba(255,255,255,0.93);
    text-shadow: 0 16px 45px rgba(0,0,0,0.35);
}

/* -------------------------
   Settings Card
-------------------------- */
.settings-card {
    background: linear-gradient(135deg, rgba(0,245,255,0.10), rgba(255,0,255,0.08));
    border-radius: 16px;
    padding: 1.0rem 1.05rem;
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 18px 55px rgba(0,0,0,0.28);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
}

/* -------------------------
   Expander (Neon Header)
-------------------------- */
div[data-testid="stExpander"] details summary {
    background: linear-gradient(135deg, rgba(0,245,255,0.18), rgba(255,0,255,0.16)) !important;
    color: rgba(255,255,255,0.92) !important;
    border-radius: 14px !important;
    padding: 0.85rem 1rem !important;
    font-size: 1.1rem !important;
    font-weight: 900 !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    box-shadow: 0 18px 55px rgba(0,0,0,0.25);
}
div[data-testid="stExpander"] details > div {
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-top: none !important;
    border-radius: 0 0 14px 14px !important;
    background: rgba(255,255,255,0.05) !important;
    backdrop-filter: blur(12px);
}

/* -------------------------
   Buttons (Ultra Neon Gradient)
-------------------------- */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #00f5ff, #ff00ff, #ffd000, #00f5ff);
    background-size: 250% 250%;
    color: rgba(255,255,255,0.98) !important;
    font-size: 1.25rem;
    font-weight: 900 !important;
    padding: 1.05rem 2.2rem;
    border-radius: 999px;
    border: 0 !important;
    cursor: pointer;
    letter-spacing: 2px;
    text-transform: uppercase;
    box-shadow:
        0 18px 60px rgba(0,0,0,0.32),
        0 0 26px rgba(0,245,255,0.22),
        0 0 26px rgba(255,0,255,0.18);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    animation: btnShift 2.2s ease-in-out infinite;
    position: relative;
    overflow: hidden;
}
@keyframes btnShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.stButton>button:hover {
    transform: translateY(-3px) scale(1.01);
    box-shadow:
        0 26px 85px rgba(0,0,0,0.42),
        0 0 42px rgba(0,245,255,0.30),
        0 0 42px rgba(255,0,255,0.26);
}
.stButton>button::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,0.35), transparent);
    transform: translateX(-120%);
    transition: transform 0.7s ease;
}
.stButton>button:hover::before { transform: translateX(120%); }

/* Download buttons */
.stDownloadButton>button {
    background: linear-gradient(90deg, #20c997, #28a745, #00f5ff) !important;
    animation: btnShift 2.2s ease-in-out infinite !important;
}

/* Make text generally readable on dark */
p, li, label, small, span, div { color: rgba(255,255,255,0.82); }
h1, h2, h3 { color: rgba(255,255,255,0.92); }

/* Streamlit default markdown separators */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent);
    margin: 1.2rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
HF_REPO_ID = "IamPradeep/Employee-Churn-Predictor"
MODEL_FILENAME = "final_random_forest_model.joblib"

BEST_FEATURES = [
    "satisfaction_level",
    "time_spend_company",
    "average_monthly_hours",
    "number_project",
    "last_evaluation",
]

FEATURE_DESCRIPTIONS = {
    "satisfaction_level": "Employee satisfaction level (0.0 - 1.0)",
    "time_spend_company": "Years at company (integer)",
    "average_monthly_hours": "Average monthly hours worked (integer)",
    "number_project": "Number of projects (integer)",
    "last_evaluation": "Last performance evaluation score (0.0 - 1.0)",
}

# =============================================================================
# LOAD MODEL FROM HUGGING FACE
# =============================================================================
@st.cache_resource
def load_model_from_huggingface():
    """Load the trained model from Hugging Face Hub"""
    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


# =============================================================================
# SESSION STATE INITIALIZATION + SYNC CALLBACKS
# =============================================================================
def init_state():
    # Canonical values
    if "satisfaction_level" not in st.session_state:
        st.session_state.satisfaction_level = 0.50
    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = 0.70

    # Widget keys mirrored to canonical values (prevents desync)
    if "sat_slider" not in st.session_state:
        st.session_state.sat_slider = float(st.session_state.satisfaction_level)
    if "sat_input" not in st.session_state:
        st.session_state.sat_input = float(st.session_state.satisfaction_level)

    if "eval_slider" not in st.session_state:
        st.session_state.eval_slider = float(st.session_state.last_evaluation)
    if "eval_input" not in st.session_state:
        st.session_state.eval_input = float(st.session_state.last_evaluation)


def sync_satisfaction_from_slider():
    v = float(st.session_state.sat_slider)
    st.session_state.satisfaction_level = v
    st.session_state.sat_input = v


def sync_satisfaction_from_input():
    v = float(st.session_state.sat_input)
    st.session_state.satisfaction_level = v
    st.session_state.sat_slider = v


def sync_evaluation_from_slider():
    v = float(st.session_state.eval_slider)
    st.session_state.last_evaluation = v
    st.session_state.eval_input = v


def sync_evaluation_from_input():
    v = float(st.session_state.eval_input)
    st.session_state.last_evaluation = v
    st.session_state.eval_slider = v


# =============================================================================
# HELPERS
# =============================================================================
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def convert_df_to_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
    return output.getvalue()


# =============================================================================
# INDIVIDUAL PREDICTION TAB
# =============================================================================
def render_individual_prediction_tab(model):
    init_state()

    st.markdown('<h2 class="section-header">📝 Enter Employee Information</h2>', unsafe_allow_html=True)

    # ROW 1
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <span style="font-size: 1.15rem;">😊 <strong>Satisfaction Level</strong></span>
                <div style="opacity:0.8; margin-top:6px; font-size:0.9rem;">0.00 (low) → 1.00 (high)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sat_col1, sat_col2 = st.columns([3, 1])
        with sat_col1:
            st.slider(
                "Satisfaction Slider",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.sat_slider),
                step=0.01,
                label_visibility="collapsed",
                key="sat_slider",
                on_change=sync_satisfaction_from_slider,
            )
        with sat_col2:
            st.number_input(
                "Satisfaction Input",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.sat_input),
                step=0.01,
                format="%.2f",
                label_visibility="collapsed",
                key="sat_input",
                on_change=sync_satisfaction_from_input,
            )

        satisfaction_level = float(st.session_state.satisfaction_level)

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <span style="font-size: 1.15rem;">📊 <strong>Last Evaluation</strong></span>
                <div style="opacity:0.8; margin-top:6px; font-size:0.9rem;">0.00 (poor) → 1.00 (excellent)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        eval_col1, eval_col2 = st.columns([3, 1])
        with eval_col1:
            st.slider(
                "Evaluation Slider",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.eval_slider),
                step=0.01,
                label_visibility="collapsed",
                key="eval_slider",
                on_change=sync_evaluation_from_slider,
            )
        with eval_col2:
            st.number_input(
                "Evaluation Input",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.eval_input),
                step=0.01,
                format="%.2f",
                label_visibility="collapsed",
                key="eval_input",
                on_change=sync_evaluation_from_input,
            )

        last_evaluation = float(st.session_state.last_evaluation)

    # ROW 2
    col3, col4, col5 = st.columns(3)

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <span style="font-size: 1.1rem;">📅 <strong>Years at Company</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time_spend_company = st.number_input(
            "Years",
            min_value=1,
            max_value=40,
            value=3,
            step=1,
            label_visibility="collapsed",
            key="individual_years",
        )

    with col4:
        st.markdown(
            """
            <div class="feature-card">
                <span style="font-size: 1.1rem;">📁 <strong>Number of Projects</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        number_project = st.number_input(
            "Projects",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            label_visibility="collapsed",
            key="individual_projects",
        )

    with col5:
        st.markdown(
            """
            <div class="feature-card">
                <span style="font-size: 1.1rem;">⏰ <strong>Avg. Monthly Hours</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        average_monthly_hours = st.number_input(
            "Hours",
            min_value=80,
            max_value=350,
            value=200,
            step=5,
            label_visibility="collapsed",
            key="individual_hours",
        )

    input_data = {
        "satisfaction_level": satisfaction_level,
        "time_spend_company": time_spend_company,
        "average_monthly_hours": average_monthly_hours,
        "number_project": number_project,
        "last_evaluation": last_evaluation,
    }

    st.markdown("---")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        predict_button = st.button("🔮 Predict Employee Turnover", use_container_width=True, key="individual_predict")

    if predict_button:
        input_df = pd.DataFrame([input_data])[BEST_FEATURES]

        prediction = int(model.predict(input_df)[0])
        prediction_proba = model.predict_proba(input_df)[0]

        prob_stay = float(prediction_proba[0]) * 100
        prob_leave = float(prediction_proba[1]) * 100

        st.markdown("---")
        st.subheader("🎯 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 0:
                st.markdown(
                    f"""
                    <div class="prediction-box stay-prediction">
                        <h1 style="margin:0; font-weight:900;">✅ STAY</h1>
                        <p style="font-size: 1.1rem; margin-top: 0.8rem;">
                            Employee is likely to <strong>STAY</strong> with the company
                        </p>
                        <div style="margin-top:0.5rem; opacity:0.85;">Confidence shines when the risk stays low.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="prediction-box leave-prediction">
                        <h1 style="margin:0; font-weight:900;">⚠️ LEAVE</h1>
                        <p style="font-size: 1.1rem; margin-top: 0.8rem;">
                            Employee is likely to <strong>LEAVE</strong> the company
                        </p>
                        <div style="margin-top:0.5rem; opacity:0.85;">Consider proactive retention actions.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col2:
            st.markdown("### 📊 Prediction Probabilities")

            st.write(f"**Probability of Staying:** {prob_stay:.1f}%")
            st.markdown(
                f"""
                <div class="progress-bar-container">
                    <div class="progress-bar-green" style="width: {prob_stay}%;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(f"**Probability of Leaving:** {prob_leave:.1f}%")
            st.markdown(
                f"""
                <div class="progress-bar-container">
                    <div class="progress-bar-red" style="width: {prob_leave}%;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# BATCH PREDICTION TAB
# =============================================================================
def render_batch_prediction_tab(model):
    st.markdown('<h2 class="section-header">📊 Batch Employee Prediction</h2>', unsafe_allow_html=True)

    with st.expander("📋 Required Columns in Your File (Click to Expand)"):
        col1, col2 = st.columns(2)
        with col1:
            for feature in BEST_FEATURES[:3]:
                st.markdown(
                    f"""
                    <div class="feature-card" style="margin-bottom:10px;">
                        <div style="font-size:1.05rem;"><strong>{feature}</strong></div>
                        <div style="opacity:0.85; font-size:0.9rem; margin-top:6px;">{FEATURE_DESCRIPTIONS[feature]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with col2:
            for feature in BEST_FEATURES[3:]:
                st.markdown(
                    f"""
                    <div class="feature-card" style="margin-bottom:10px;">
                        <div style="font-size:1.05rem;"><strong>{feature}</strong></div>
                        <div style="opacity:0.85; font-size:0.9rem; margin-top:6px;">{FEATURE_DESCRIPTIONS[feature]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.info(
            "💡 Tip: Your file can include extra columns (employee_id, department, etc.). "
            "They will be preserved in the output but ignored during prediction."
        )

    st.markdown("---")
    st.markdown("### 📁 Upload Your Data")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<div class="settings-card"><h4 style="margin:0;">⚙️ File Settings</h4></div>', unsafe_allow_html=True)
        file_format = st.selectbox(
            "Select file format",
            options=["CSV", "Excel (.xlsx)"],
            index=0,
            key="file_format",
        )

    with col2:
        if file_format == "CSV":
            uploaded_file = st.file_uploader(
                "Upload your CSV file",
                type=["csv"],
                key="csv_uploader",
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload your Excel file",
                type=["xlsx", "xls"],
                key="excel_uploader",
            )

    st.markdown("---")
    st.markdown("### ⚙️ Prediction Output Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="settings-card"><h4 style="margin:0;">📝 Column Name for Predictions</h4></div>', unsafe_allow_html=True)
        column_name_option = st.selectbox(
            "Select prediction column name",
            options=["Prediction", "Churn", "Will_Leave", "Turnover", "Custom"],
            index=0,
            key="column_name_option",
        )

        if column_name_option == "Custom":
            custom_column_name = st.text_input(
                "Enter custom column name",
                value="My_Prediction",
                key="custom_column_name",
            )
            prediction_column_name = custom_column_name.strip() or "Prediction"
        else:
            prediction_column_name = column_name_option

    with col2:
        st.markdown('<div class="settings-card"><h4 style="margin:0;">🏷️ Prediction Labels</h4></div>', unsafe_allow_html=True)

        label_option = st.selectbox(
            "Select prediction labels",
            options=["Leave / Stay", "Yes / No", "Churn / Not Churn", "1 / 0", "True / False", "Custom"],
            index=0,
            key="label_option",
        )

        label_mappings = {
            "Leave / Stay": {1: "Leave", 0: "Stay"},
            "Yes / No": {1: "Yes", 0: "No"},
            "Churn / Not Churn": {1: "Churn", 0: "Not Churn"},
            "1 / 0": {1: "1", 0: "0"},
            "True / False": {1: "True", 0: "False"},
        }

        if label_option == "Custom":
            c1, c2 = st.columns(2)
            with c1:
                custom_leave_label = st.text_input(
                    "Label for LEAVING",
                    value="Leaving",
                    key="custom_leave_label",
                )
            with c2:
                custom_stay_label = st.text_input(
                    "Label for STAYING",
                    value="Staying",
                    key="custom_stay_label",
                )
            prediction_labels = {
                1: custom_leave_label.strip() or "Leaving",
                0: custom_stay_label.strip() or "Staying",
            }
        else:
            prediction_labels = label_mappings[label_option]

        st.info(f"📌 Label Preview: Leaving → '{prediction_labels[1]}' | Staying → '{prediction_labels[0]}'")

    st.markdown("---")
    st.markdown("### 📊 Additional Output Options")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="settings-card"><h4 style="margin:0;">🎯 Probability Columns</h4></div>', unsafe_allow_html=True)
        include_probabilities = st.checkbox(
            "Include prediction probabilities in output",
            value=True,
            key="include_probabilities",
        )

    with col2:
        st.markdown('<div class="settings-card"><h4 style="margin:0;">⚠️ High Risk Filter</h4></div>', unsafe_allow_html=True)
        include_high_risk_download = st.checkbox(
            "Enable high-risk employees download",
            value=True,
            key="include_high_risk",
        )

    if uploaded_file is not None:
        st.markdown("---")
        st.markdown("### 📄 Uploaded Data Preview")

        try:
            if file_format == "CSV":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f"""
                    <div class="stats-card">
                        <h4>Total Rows</h4>
                        <div class="number">{len(df):,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"""
                    <div class="stats-card">
                        <h4>Total Columns</h4>
                        <div class="number">{len(df.columns):,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col3:
                available_features = [col for col in BEST_FEATURES if col in df.columns]
                st.markdown(
                    f"""
                    <div class="stats-card">
                        <h4>Required Cols Found</h4>
                        <div class="number">{len(available_features)}/{len(BEST_FEATURES)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.dataframe(df.head(12), use_container_width=True)

            missing_columns = [col for col in BEST_FEATURES if col not in df.columns]
            if missing_columns:
                st.error(
                    "Missing required columns: " + ", ".join(missing_columns)
                    + "\n\nMake sure your column names match exactly."
                )
                return

            with st.expander("🔍 View columns being used for prediction"):
                for feature in BEST_FEATURES:
                    sample_values = df[feature].head(3).tolist()
                    st.write(f"• **{feature}** → sample values: `{sample_values}`")

            st.markdown("---")
            _, mid, _ = st.columns([1, 2, 1])
            with mid:
                batch_predict_button = st.button(
                    "🔮 Generate Batch Predictions",
                    use_container_width=True,
                    key="batch_predict",
                )

            if batch_predict_button:
                with st.spinner("Processing predictions..."):
                    input_features = df[BEST_FEATURES].copy()
                    predictions = model.predict(input_features).astype(int)
                    prediction_probabilities = model.predict_proba(input_features)

                    result_df = df.copy()
                    result_df[prediction_column_name] = [prediction_labels[p] for p in predictions]

                    if include_probabilities:
                        result_df[f"{prediction_column_name}_Probability_Stay"] = (prediction_probabilities[:, 0] * 100).round(2)
                        result_df[f"{prediction_column_name}_Probability_Leave"] = (prediction_probabilities[:, 1] * 100).round(2)

                st.success("✅ Predictions generated successfully!")

                st.markdown("---")
                st.markdown("### 📊 Prediction Results")

                leaving_count = int((predictions == 1).sum())
                staying_count = int((predictions == 0).sum())
                leaving_percentage = leaving_count / len(predictions) * 100
                staying_percentage = staying_count / len(predictions) * 100
                avg_leave_prob = float(prediction_probabilities[:, 1].mean() * 100)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(
                        f"""
                        <div class="stats-card">
                            <h4>Total Employees</h4>
                            <div class="number">{len(predictions):,}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        f"""
                        <div class="stats-card" style="border:1px solid rgba(220,53,69,0.35);">
                            <h4>Predicted to Leave</h4>
                            <div class="number" style="color: rgba(255,120,140,0.95);">{leaving_count:,}</div>
                            <div style="opacity:0.8;">({leaving_percentage:.1f}%)</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col3:
                    st.markdown(
                        f"""
                        <div class="stats-card" style="border:1px solid rgba(40,167,69,0.35);">
                            <h4>Predicted to Stay</h4>
                            <div class="number" style="color: rgba(140,255,200,0.95);">{staying_count:,}</div>
                            <div style="opacity:0.8;">({staying_percentage:.1f}%)</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col4:
                    st.markdown(
                        f"""
                        <div class="stats-card" style="border:1px solid rgba(255,208,0,0.30);">
                            <h4>Avg. Leave Probability</h4>
                            <div class="number" style="color: rgba(255,208,0,0.95);">{avg_leave_prob:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("#### 📈 Turnover Distribution")
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Staying:** {staying_percentage:.1f}%")
                    st.markdown(
                        f"""
                        <div class="progress-bar-container">
                            <div class="progress-bar-green" style="width: {staying_percentage}%;"></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.write(f"**Leaving:** {leaving_percentage:.1f}%")
                    st.markdown(
                        f"""
                        <div class="progress-bar-container">
                            <div class="progress-bar-red" style="width: {leaving_percentage}%;"></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("#### 📄 Result Data Preview")
                st.dataframe(result_df.head(20), use_container_width=True)

                st.markdown("---")
                st.markdown("### 📥 Download Results")

                if include_high_risk_download:
                    d1, d2, d3 = st.columns([1, 1, 1])
                else:
                    d1, d2 = st.columns([1, 1])

                with d1:
                    st.download_button(
                        label="📥 Download as CSV",
                        data=convert_df_to_csv(result_df),
                        file_name="employee_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_csv",
                    )

                with d2:
                    st.download_button(
                        label="📥 Download as Excel",
                        data=convert_df_to_excel(result_df),
                        file_name="employee_predictions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="download_excel",
                    )

                if include_high_risk_download:
                    with d3:
                        mask = prediction_probabilities[:, 1] > 0.5
                        high_risk_df = result_df.loc[mask].copy()

                        if len(high_risk_df) > 0:
                            st.download_button(
                                label=f"📥 High Risk Only ({len(high_risk_df)})",
                                data=convert_df_to_csv(high_risk_df),
                                file_name="high_risk_employees.csv",
                                mime="text/csv",
                                use_container_width=True,
                                key="download_high_risk",
                            )
                        else:
                            st.info("No high-risk employees found (> 50% leave probability).")

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please ensure the file is valid and not corrupted.")

    else:
        st.markdown("---")
        with st.expander("📋 View Sample Data Format"):
            sample_data = pd.DataFrame(
                {
                    "employee_id": [1, 2, 3, 4, 5],
                    "satisfaction_level": [0.38, 0.80, 0.11, 0.72, 0.37],
                    "time_spend_company": [3, 5, 4, 3, 2],
                    "average_monthly_hours": [157, 262, 272, 223, 159],
                    "number_project": [2, 5, 7, 5, 2],
                    "last_evaluation": [0.53, 0.86, 0.88, 0.87, 0.52],
                    "department": ["sales", "IT", "IT", "sales", "hr"],
                    "salary": ["low", "medium", "medium", "high", "low"],
                }
            )
            st.dataframe(sample_data, use_container_width=True)
            st.info("Only the 5 required columns are used for prediction; extra columns are preserved.")


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    st.markdown(
        """
        <div class="hero-wrap">
          <div class="hero">
            <div>
              <h1 class="hero-title">👥 Employee <span>Turnover</span> Predictor</h1>
              <div class="hero-sub">
                A high-impact, ultra-polished interface for predicting whether employees are likely to stay or leave —
                with both individual and batch workflows.
              </div>
              <div class="hero-badges">
                <div class="badge">⚡ Fast Predictions</div>
                <div class="badge">🧠 Random Forest Model</div>
                <div class="badge">📦 Batch Upload + Export</div>
                <div class="badge">🎯 Probabilities Included</div>
              </div>
            </div>
            <div class="hero-right">
              <div class="kpi">
                <div class="kpi-card">
                  <div class="kpi-top">Model Input Features</div>
                  <div class="kpi-val">5</div>
                  <div class="kpi-foot">Optimized feature set</div>
                </div>
                <div class="kpi-card">
                  <div class="kpi-top">Modes</div>
                  <div class="kpi-val">2</div>
                  <div class="kpi-foot">Individual + Batch</div>
                </div>
                <div class="kpi-card">
                  <div class="kpi-top">Outputs</div>
                  <div class="kpi-val">CSV / XLSX</div>
                  <div class="kpi-foot">Ready to share</div>
                </div>
                <div class="kpi-card">
                  <div class="kpi-top">Risk Filter</div>
                  <div class="kpi-val">✓</div>
                  <div class="kpi-foot">High-risk export</div>
                </div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = load_model_from_huggingface()
    if model is None:
        st.error("❌ Failed to load model. Please check the Hugging Face repository.")
        st.info(f"Repository: https://huggingface.co/{HF_REPO_ID}")
        return

    tab1, tab2 = st.tabs(["📝 Individual Prediction", "📊 Batch Prediction"])
    with tab1:
        render_individual_prediction_tab(model)
    with tab2:
        render_batch_prediction_tab(model)

    st.markdown(
        """
        <div style="text-align:center; opacity:0.75; margin-top: 2rem; padding-bottom: 1.5rem;">
          <small>Built with Streamlit • Ultra-Glass UI • Predict responsibly (use as decision support, not as a sole decision-maker).</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
