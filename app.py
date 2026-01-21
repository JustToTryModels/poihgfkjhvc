import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import io

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Employee Turnover Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# ULTRA DECORATIVE • JAW-DROPPING UI (CUSTOM CSS)
# ============================================================================
st.markdown(
    """
<style>
/* =========================
   GLOBAL THEME (AURORA / NEON GLASS)
   ========================= */
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;700;900&family=Poppins:wght@400;600;800;900&display=swap');

:root{
  --ink:#061425;
  --ink2:#0b2240;
  --glass: rgba(255,255,255,0.10);
  --glass2: rgba(255,255,255,0.07);
  --stroke: rgba(255,255,255,0.18);
  --stroke2: rgba(255,255,255,0.10);

  --a1:#7c3aed;
  --a2:#06b6d4;
  --a3:#22c55e;
  --a4:#f59e0b;
  --a5:#ef4444;

  --good:#22c55e;
  --bad:#ef4444;
  --warn:#f59e0b;
}

html, body, [class*="css"]{
  font-family: "Poppins", "Montserrat", system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
}

/* Full-page animated aurora background */
.stApp{
  background:
    radial-gradient(1200px 800px at 10% 10%, rgba(124,58,237,0.45), transparent 55%),
    radial-gradient(900px 700px at 90% 20%, rgba(6,182,212,0.40), transparent 55%),
    radial-gradient(900px 700px at 30% 90%, rgba(34,197,94,0.25), transparent 60%),
    radial-gradient(700px 600px at 85% 85%, rgba(245,158,11,0.25), transparent 58%),
    linear-gradient(135deg, #050812 0%, #07162a 40%, #04060f 100%);
  position: relative;
  overflow-x: hidden;
}

/* Floating animated blobs */
.stApp:before, .stApp:after{
  content:"";
  position: fixed;
  z-index: 0;
  width: 520px;
  height: 520px;
  filter: blur(35px);
  opacity: 0.55;
  border-radius: 50%;
  pointer-events: none;
  mix-blend-mode: screen;
  animation: floaty 10s ease-in-out infinite;
}
.stApp:before{
  left: -120px; top: 120px;
  background: radial-gradient(circle at 30% 30%, rgba(124,58,237,0.75), rgba(6,182,212,0.25), transparent 60%);
}
.stApp:after{
  right: -160px; bottom: 80px;
  background: radial-gradient(circle at 40% 40%, rgba(239,68,68,0.65), rgba(245,158,11,0.25), transparent 60%);
  animation-delay: -4s;
}
@keyframes floaty{
  0%{ transform: translate(0,0) scale(1); }
  50%{ transform: translate(30px,-20px) scale(1.06); }
  100%{ transform: translate(0,0) scale(1); }
}

/* Center content with glass panel */
.block-container{
  max-width: 1220px !important;
  padding-left: 3.2rem !important;
  padding-right: 3.2rem !important;
  padding-top: 2.0rem !important;
  padding-bottom: 3.0rem !important;
  position: relative;
  z-index: 1;
}

/* Add subtle glass frame around entire main content */
main .block-container{
  background: linear-gradient(180deg, var(--glass), rgba(255,255,255,0.04));
  border: 1px solid var(--stroke2);
  box-shadow:
    0 20px 80px rgba(0,0,0,0.45),
    inset 0 1px 0 rgba(255,255,255,0.10);
  border-radius: 26px;
  backdrop-filter: blur(14px);
}

/* Remove Streamlit default top padding spacing a bit */
div[data-testid="stHeader"]{
  background: transparent;
}

/* =========================
   HERO HEADER (GRADIENT TEXT + BADGES)
   ========================= */
.hero-wrap{
  padding: 1.8rem 1.6rem 1.1rem 1.6rem;
  border-radius: 22px;
  border: 1px solid rgba(255,255,255,0.14);
  background:
    radial-gradient(600px 200px at 10% 0%, rgba(124,58,237,0.25), transparent 60%),
    radial-gradient(600px 200px at 90% 10%, rgba(6,182,212,0.20), transparent 60%),
    linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.06));
  box-shadow:
    0 18px 55px rgba(0,0,0,0.40),
    inset 0 1px 0 rgba(255,255,255,0.10);
  position: relative;
  overflow: hidden;
}
.hero-wrap:before{
  content:"";
  position:absolute;
  inset:-2px;
  background: conic-gradient(from 180deg,
    rgba(124,58,237,0.55),
    rgba(6,182,212,0.45),
    rgba(34,197,94,0.35),
    rgba(245,158,11,0.45),
    rgba(239,68,68,0.40),
    rgba(124,58,237,0.55)
  );
  filter: blur(18px);
  opacity: 0.45;
  z-index: 0;
  animation: spinGlow 8s linear infinite;
}
@keyframes spinGlow{
  0%{ transform: rotate(0deg); }
  100%{ transform: rotate(360deg); }
}
.hero-inner{
  position: relative;
  z-index: 1;
}
.hero-title{
  font-family: "Montserrat", "Poppins", sans-serif !important;
  font-weight: 900;
  font-size: 2.8rem;
  line-height: 1.05;
  text-align: center;
  margin: 0.2rem 0 0.35rem 0;
  background: linear-gradient(90deg, #ffffff 0%, #bcd7ff 25%, #9ff7ff 50%, #ffe6b0 75%, #ffffff 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  text-shadow: 0 12px 50px rgba(0,0,0,0.35);
}
.hero-sub{
  text-align:center;
  color: rgba(255,255,255,0.80);
  font-size: 1.05rem;
  margin: 0 0 0.9rem 0;
}
.badges{
  display:flex;
  gap: 10px;
  justify-content:center;
  flex-wrap: wrap;
}
.badge{
  padding: 0.45rem 0.75rem;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.08);
  color: rgba(255,255,255,0.88);
  font-size: 0.9rem;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.12);
}
.badge strong{ color:white; }

/* =========================
   SECTION TITLES
   ========================= */
.section-header{
  font-family: "Montserrat", "Poppins", sans-serif !important;
  font-weight: 900;
  font-size: 1.55rem;
  text-align: center;
  color: rgba(255,255,255,0.92);
  margin: 0.8rem 0 1.2rem 0;
  letter-spacing: 0.2px;
}
.stMarkdown hr{
  border: none;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent);
  margin: 1.25rem 0;
}

/* =========================
   SUPER TABS (NEON / GLASS)
   ========================= */
.stTabs [data-baseweb="tab-list"]{
  gap: 18px;
  justify-content: center;
  background-color: transparent;
  padding: 6px 0 0 0;
}
.stTabs [data-baseweb="tab"]{
  height: 86px !important;
  min-width: 360px !important;
  padding: 0 52px !important;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.06));
  border-radius: 18px 18px 18px 18px !important;
  font-weight: 900 !important;
  font-size: 1.35rem !important;
  color: rgba(255,255,255,0.85) !important;
  border: 1px solid rgba(255,255,255,0.16) !important;
  transition: transform 0.25s ease, box-shadow 0.25s ease, background 0.25s ease !important;
  display:flex !important;
  align-items:center !important;
  justify-content:center !important;
  backdrop-filter: blur(10px);
  position: relative;
  overflow: hidden;
}
.stTabs [data-baseweb="tab"]::before{
  content:"";
  position:absolute;
  inset:-2px;
  background: conic-gradient(from 180deg,
    rgba(124,58,237,0.45),
    rgba(6,182,212,0.35),
    rgba(34,197,94,0.28),
    rgba(245,158,11,0.35),
    rgba(239,68,68,0.28),
    rgba(124,58,237,0.45)
  );
  filter: blur(18px);
  opacity: 0.0;
  transition: opacity 0.25s ease;
}
.stTabs [data-baseweb="tab"]:hover{
  transform: translateY(-4px);
  box-shadow: 0 18px 45px rgba(0,0,0,0.35);
}
.stTabs [data-baseweb="tab"]:hover::before{ opacity: 0.5; }

.stTabs [aria-selected="true"]{
  background:
    radial-gradient(450px 200px at 20% 0%, rgba(124,58,237,0.35), transparent 60%),
    radial-gradient(450px 200px at 80% 10%, rgba(6,182,212,0.30), transparent 60%),
    linear-gradient(180deg, rgba(255,255,255,0.12), rgba(255,255,255,0.07)) !important;
  border: 1px solid rgba(255,255,255,0.22) !important;
  box-shadow:
    0 22px 55px rgba(0,0,0,0.45),
    inset 0 1px 0 rgba(255,255,255,0.15) !important;
  color: white !important;
}
.stTabs [aria-selected="true"]::before{ opacity: 0.65; }

.stTabs [data-baseweb="tab-border"],
.stTabs [data-baseweb="tab-highlight"]{
  display:none !important;
}

/* =========================
   FEATURE CARDS (NEON GLASS)
   ========================= */
.feature-card{
  background:
    radial-gradient(500px 130px at 10% 0%, rgba(6,182,212,0.18), transparent 60%),
    radial-gradient(500px 130px at 90% 0%, rgba(124,58,237,0.18), transparent 60%),
    linear-gradient(180deg, rgba(255,255,255,0.11), rgba(255,255,255,0.06));
  border-radius: 18px;
  padding: 1.05rem 1.15rem;
  border: 1px solid rgba(255,255,255,0.18);
  box-shadow:
    0 16px 40px rgba(0,0,0,0.35),
    inset 0 1px 0 rgba(255,255,255,0.12);
  position: relative;
  overflow: hidden;
  margin-bottom: 0.9rem;
}
.feature-card::after{
  content:"";
  position:absolute;
  top:-80px; left:-120px;
  width: 220px; height: 220px;
  background: radial-gradient(circle, rgba(255,255,255,0.22), transparent 60%);
  transform: rotate(25deg);
  opacity: 0.25;
}
.feature-card .title{
  color: rgba(255,255,255,0.92);
  font-weight: 900;
  font-size: 1.15rem;
  letter-spacing: 0.2px;
}
.feature-card .hint{
  color: rgba(255,255,255,0.70);
  font-size: 0.95rem;
  margin-top: 0.25rem;
}

/* Inputs styling (text/number) */
.stNumberInput input, .stTextInput input{
  background: rgba(255,255,255,0.08) !important;
  border: 1px solid rgba(255,255,255,0.16) !important;
  color: rgba(255,255,255,0.92) !important;
  border-radius: 12px !important;
}
.stNumberInput input:focus, .stTextInput input:focus{
  border: 1px solid rgba(6,182,212,0.55) !important;
  box-shadow: 0 0 0 4px rgba(6,182,212,0.12) !important;
}

/* Selectbox */
div[data-baseweb="select"] > div{
  background: rgba(255,255,255,0.08) !important;
  border: 1px solid rgba(255,255,255,0.16) !important;
  border-radius: 12px !important;
}
div[data-baseweb="select"] *{
  color: rgba(255,255,255,0.92) !important;
}

/* File uploader */
section[data-testid="stFileUploaderDropzone"]{
  background: rgba(255,255,255,0.07) !important;
  border: 1.5px dashed rgba(255,255,255,0.22) !important;
  border-radius: 16px !important;
}
section[data-testid="stFileUploaderDropzone"] *{
  color: rgba(255,255,255,0.86) !important;
}

/* Checkbox */
.stCheckbox label{
  color: rgba(255,255,255,0.88) !important;
}

/* =========================
   MEGA BUTTON (NEON + SHIMMER)
   ========================= */
.stButton>button{
  width:100%;
  border:none !important;
  border-radius: 999px !important;
  padding: 1.15rem 2.2rem !important;
  font-weight: 900 !important;
  font-size: 1.25rem !important;
  letter-spacing: 2.5px !important;
  text-transform: uppercase;

  color: white !important;

  background: linear-gradient(45deg, #7c3aed, #06b6d4, #22c55e, #f59e0b, #ef4444, #7c3aed);
  background-size: 420% 420%;
  animation: gradientShift 2.8s ease infinite, buttonPulse 2.0s ease-in-out infinite;

  box-shadow:
    0 16px 45px rgba(0,0,0,0.45),
    0 0 60px rgba(124,58,237,0.20),
    0 0 80px rgba(6,182,212,0.14);
  position: relative;
  overflow: hidden;
}
@keyframes gradientShift{
  0%{ background-position: 0% 50%; }
  50%{ background-position: 100% 50%; }
  100%{ background-position: 0% 50%; }
}
@keyframes buttonPulse{
  0%{ transform: translateY(0px) scale(1); }
  50%{ transform: translateY(-2px) scale(1.02); }
  100%{ transform: translateY(0px) scale(1); }
}
.stButton>button::before{
  content:"";
  position:absolute;
  top:0; left:-120%;
  width: 120%;
  height: 100%;
  background: linear-gradient(120deg, transparent, rgba(255,255,255,0.45), transparent);
  transform: skewX(-18deg);
  transition: left 0.65s ease;
}
.stButton>button:hover::before{ left: 120%; }
.stButton>button:hover{
  filter: saturate(1.15);
  box-shadow:
    0 24px 70px rgba(0,0,0,0.55),
    0 0 80px rgba(6,182,212,0.24),
    0 0 110px rgba(239,68,68,0.12);
  transform: translateY(-4px) scale(1.02);
}
.stButton>button:active{
  transform: translateY(1px) scale(0.99);
}
.stButton>button:focus,
.stButton>button:focus-visible{
  outline: none !important;
  box-shadow:
    0 24px 70px rgba(0,0,0,0.55),
    0 0 0 4px rgba(6,182,212,0.18),
    0 0 90px rgba(124,58,237,0.20);
}

/* Download button styling */
.stDownloadButton>button{
  background: linear-gradient(135deg, rgba(34,197,94,1), rgba(6,182,212,1)) !important;
  background-size: 200% 200% !important;
  animation: gradientShift 2.4s ease infinite !important;
  letter-spacing: 1px !important;
}

/* =========================
   PREDICTION BOXES (GLOW / NEON BORDER)
   ========================= */
.prediction-shell{
  border-radius: 20px;
  padding: 1.2rem;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.06);
  box-shadow: 0 22px 65px rgba(0,0,0,0.45);
  position: relative;
  overflow:hidden;
}
.prediction-shell::before{
  content:"";
  position:absolute;
  inset:-2px;
  background: conic-gradient(from 180deg,
    rgba(124,58,237,0.55),
    rgba(6,182,212,0.45),
    rgba(34,197,94,0.40),
    rgba(245,158,11,0.45),
    rgba(239,68,68,0.38),
    rgba(124,58,237,0.55)
  );
  filter: blur(18px);
  opacity: 0.35;
  animation: spinGlow 7s linear infinite;
}
.prediction-inner{
  position:relative;
  z-index:1;
  border-radius: 16px;
  padding: 1.5rem;
  background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.05));
  border: 1px solid rgba(255,255,255,0.14);
}
.big-label{
  font-family: "Montserrat", "Poppins", sans-serif !important;
  font-weight: 900;
  font-size: 2.4rem;
  margin: 0;
  text-align:center;
  letter-spacing: 0.5px;
  color: white;
  text-shadow: 0 18px 55px rgba(0,0,0,0.55);
}
.sub-label{
  margin: 0.55rem 0 0 0;
  font-size: 1.05rem;
  text-align:center;
  color: rgba(255,255,255,0.84);
}
.tag-stay{
  display:inline-block;
  margin-top: 0.75rem;
  padding: 0.35rem 0.65rem;
  border-radius: 999px;
  background: rgba(34,197,94,0.18);
  border: 1px solid rgba(34,197,94,0.35);
  color: rgba(255,255,255,0.92);
  font-weight: 800;
}
.tag-leave{
  display:inline-block;
  margin-top: 0.75rem;
  padding: 0.35rem 0.65rem;
  border-radius: 999px;
  background: rgba(239,68,68,0.18);
  border: 1px solid rgba(239,68,68,0.35);
  color: rgba(255,255,255,0.92);
  font-weight: 800;
}

/* Progress bars */
.progress-wrap{
  width: 100%;
  height: 16px;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 999px;
  overflow:hidden;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.10);
}
.progress-fill{
  height: 100%;
  border-radius: 999px;
  transition: width 0.6s ease;
}
.progress-good{
  background: linear-gradient(90deg, rgba(34,197,94,0.95), rgba(6,182,212,0.85));
  box-shadow: 0 0 30px rgba(34,197,94,0.25);
}
.progress-bad{
  background: linear-gradient(90deg, rgba(239,68,68,0.95), rgba(245,158,11,0.85));
  box-shadow: 0 0 30px rgba(239,68,68,0.22);
}

/* Stats cards */
.stats-card{
  background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.05));
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 18px;
  padding: 1.15rem 1rem;
  box-shadow: 0 18px 45px rgba(0,0,0,0.35);
  text-align: center;
}
.stats-card h4{
  margin: 0 0 0.35rem 0;
  color: rgba(255,255,255,0.78);
  font-weight: 700;
}
.stats-card .number{
  font-family: "Montserrat", "Poppins", sans-serif !important;
  font-weight: 900;
  font-size: 2.0rem;
  color: rgba(255,255,255,0.95);
}

/* Expander styling (lux) */
div[data-testid="stExpander"] details summary{
  background: linear-gradient(135deg, rgba(124,58,237,0.70), rgba(6,182,212,0.55)) !important;
  color: white !important;
  border-radius: 14px !important;
  padding: 0.85rem 1rem !important;
  border: 1px solid rgba(255,255,255,0.16) !important;
  box-shadow: 0 18px 45px rgba(0,0,0,0.35) !important;
}
div[data-testid="stExpander"] details > div{
  border: 1px solid rgba(255,255,255,0.14) !important;
  border-top: none !important;
  border-radius: 0 0 14px 14px !important;
  background: rgba(255,255,255,0.06) !important;
  backdrop-filter: blur(10px);
}

/* Dataframe container */
div[data-testid="stDataFrame"]{
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.14);
  box-shadow: 0 18px 55px rgba(0,0,0,0.35);
}

/* Make typical markdown text white-ish */
.stMarkdown, .stText, .stCaption, .stAlert{
  color: rgba(255,255,255,0.86);
}

/* Hide Streamlit footer */
footer{ visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
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

# ============================================================================
# LOAD MODEL FROM HUGGING FACE
# ============================================================================
@st.cache_resource
def load_model_from_huggingface():
    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model",
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# ============================================================================
# CONFETTI (for extra WOW)
# ============================================================================
def fire_confetti():
    components.html(
        """
        <canvas id="c" style="position:fixed;inset:0;pointer-events:none;z-index:9999;"></canvas>
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
        <script>
          const myCanvas = document.getElementById('c');
          const myConfetti = confetti.create(myCanvas, { resize: true, useWorker: true });
          const duration = 1200;
          const end = Date.now() + duration;

          (function frame() {
            myConfetti({
              particleCount: 7,
              angle: 60,
              spread: 75,
              origin: { x: 0 },
              colors: ["#7c3aed","#06b6d4","#22c55e","#f59e0b","#ef4444","#ffffff"]
            });
            myConfetti({
              particleCount: 7,
              angle: 120,
              spread: 75,
              origin: { x: 1 },
              colors: ["#7c3aed","#06b6d4","#22c55e","#f59e0b","#ef4444","#ffffff"]
            });

            if (Date.now() < end) requestAnimationFrame(frame);
            else setTimeout(()=>{ myCanvas.remove(); }, 250);
          }());
        </script>
        """,
        height=0,
        width=0,
    )

# ============================================================================
# CALLBACK FUNCTIONS FOR SYNCING SLIDERS AND NUMBER INPUTS (ROBUST)
# ============================================================================
def sync_satisfaction_slider():
    v = float(st.session_state.sat_slider)
    st.session_state.satisfaction_level = v
    st.session_state.sat_input = v

def sync_satisfaction_input():
    v = float(st.session_state.sat_input)
    st.session_state.satisfaction_level = v
    st.session_state.sat_slider = v

def sync_evaluation_slider():
    v = float(st.session_state.eval_slider)
    st.session_state.last_evaluation = v
    st.session_state.eval_input = v

def sync_evaluation_input():
    v = float(st.session_state.eval_input)
    st.session_state.last_evaluation = v
    st.session_state.eval_slider = v

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def convert_df_to_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
    return output.getvalue()

# ============================================================================
# INDIVIDUAL PREDICTION TAB
# ============================================================================
def render_individual_prediction_tab(model):
    st.markdown("---")
    st.markdown('<h2 class="section-header">📝 Enter Employee Information</h2>', unsafe_allow_html=True)

    # Initialize session state (and widget keys) for perfect syncing
    if "satisfaction_level" not in st.session_state:
        st.session_state.satisfaction_level = 0.50
    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = 0.70
    if "sat_slider" not in st.session_state:
        st.session_state.sat_slider = st.session_state.satisfaction_level
    if "sat_input" not in st.session_state:
        st.session_state.sat_input = st.session_state.satisfaction_level
    if "eval_slider" not in st.session_state:
        st.session_state.eval_slider = st.session_state.last_evaluation
    if "eval_input" not in st.session_state:
        st.session_state.eval_input = st.session_state.last_evaluation

    # Row 1: satisfaction & evaluation
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
              <div class="title">😊 Satisfaction Level</div>
              <div class="hint">0.00 = very dissatisfied • 1.00 = very satisfied</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        s1, s2 = st.columns([3, 1])
        with s1:
            st.slider(
                "Satisfaction Slider",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.satisfaction_level),
                step=0.01,
                label_visibility="collapsed",
                key="sat_slider",
                on_change=sync_satisfaction_slider,
            )
        with s2:
            st.number_input(
                "Satisfaction Input",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.satisfaction_level),
                step=0.01,
                format="%.2f",
                label_visibility="collapsed",
                key="sat_input",
                on_change=sync_satisfaction_input,
            )
        satisfaction_level = float(st.session_state.satisfaction_level)

    with col2:
        st.markdown(
            """
            <div class="feature-card">
              <div class="title">📊 Last Evaluation</div>
              <div class="hint">0.00 = poor • 1.00 = excellent</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        e1, e2 = st.columns([3, 1])
        with e1:
            st.slider(
                "Evaluation Slider",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.last_evaluation),
                step=0.01,
                label_visibility="collapsed",
                key="eval_slider",
                on_change=sync_evaluation_slider,
            )
        with e2:
            st.number_input(
                "Evaluation Input",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.last_evaluation),
                step=0.01,
                format="%.2f",
                label_visibility="collapsed",
                key="eval_input",
                on_change=sync_evaluation_input,
            )
        last_evaluation = float(st.session_state.last_evaluation)

    # Row 2
    col3, col4, col5 = st.columns(3)

    with col3:
        st.markdown(
            """
            <div class="feature-card">
              <div class="title">📅 Years at Company</div>
              <div class="hint">How long the employee has been with the company</div>
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
              <div class="title">📁 Number of Projects</div>
              <div class="hint">Active projects currently assigned</div>
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
              <div class="title">⏰ Avg. Monthly Hours</div>
              <div class="hint">Average working hours per month</div>
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
        "time_spend_company": int(time_spend_company),
        "average_monthly_hours": int(average_monthly_hours),
        "number_project": int(number_project),
        "last_evaluation": last_evaluation,
    }

    st.markdown("---")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        predict_button = st.button("🔮 Predict Employee Turnover", use_container_width=True, key="individual_predict")

    if predict_button:
        input_df = pd.DataFrame([input_data])[BEST_FEATURES]
        prediction = int(model.predict(input_df)[0])
        prediction_proba = model.predict_proba(input_df)[0]

        prob_stay = float(prediction_proba[0] * 100)
        prob_leave = float(prediction_proba[1] * 100)

        fire_confetti()

        st.markdown("---")
        st.markdown('<h2 class="section-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)

        left, right = st.columns([1.05, 0.95])

        with left:
            if prediction == 0:
                st.markdown(
                    f"""
                    <div class="prediction-shell">
                      <div class="prediction-inner">
                        <p class="big-label">✅ STAY</p>
                        <p class="sub-label">Employee is likely to <strong>STAY</strong> with the company</p>
                        <div style="text-align:center;"><span class="tag-stay">Low Turnover Risk</span></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="prediction-shell">
                      <div class="prediction-inner">
                        <p class="big-label">⚠️ LEAVE</p>
                        <p class="sub-label">Employee is likely to <strong>LEAVE</strong> the company</p>
                        <div style="text-align:center;"><span class="tag-leave">High Turnover Risk</span></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with right:
            st.markdown('<h3 style="color:rgba(255,255,255,0.92); font-weight:900; margin-top:0.2rem;">📊 Probabilities</h3>', unsafe_allow_html=True)

            st.markdown(f"**Probability of Staying:** {prob_stay:.1f}%")
            st.markdown(
                f"""
                <div class="progress-wrap">
                  <div class="progress-fill progress-good" style="width:{prob_stay:.2f}%;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(f"<div style='height:10px;'></div>", unsafe_allow_html=True)

            st.markdown(f"**Probability of Leaving:** {prob_leave:.1f}%")
            st.markdown(
                f"""
                <div class="progress-wrap">
                  <div class="progress-fill progress-bad" style="width:{prob_leave:.2f}%;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ============================================================================
# BATCH PREDICTION TAB
# ============================================================================
def render_batch_prediction_tab(model):
    st.markdown("---")
    st.markdown('<h2 class="section-header">📊 Batch Employee Prediction</h2>', unsafe_allow_html=True)

    with st.expander("📋 Required Columns in Your File (Click to Expand)"):
        st.markdown(
            """
            <div class="feature-card" style="margin-bottom:0.6rem;">
              <div class="title">Your uploaded file must contain these exact column names</div>
              <div class="hint">Extra columns are allowed and will be preserved in the output.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            for feature in BEST_FEATURES[:3]:
                st.markdown(
                    f"""
                    <div class="feature-card" style="margin-bottom:0.6rem;">
                      <div class="title">• {feature}</div>
                      <div class="hint">{FEATURE_DESCRIPTIONS[feature]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with col2:
            for feature in BEST_FEATURES[3:]:
                st.markdown(
                    f"""
                    <div class="feature-card" style="margin-bottom:0.6rem;">
                      <div class="title">• {feature}</div>
                      <div class="hint">{FEATURE_DESCRIPTIONS[feature]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown('<h2 class="section-header">📁 Upload Your Data</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        file_format = st.selectbox(
            "Select file format",
            options=["CSV", "Excel (.xlsx)"],
            index=0,
            help="Choose the format of your data file",
            key="file_format",
        )
    with col2:
        if file_format == "CSV":
            uploaded_file = st.file_uploader(
                "Upload your CSV file",
                type=["csv"],
                help="Upload a CSV file containing employee data",
                key="csv_uploader",
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload your Excel file",
                type=["xlsx", "xls"],
                help="Upload an Excel file containing employee data",
                key="excel_uploader",
            )

    st.markdown("---")
    st.markdown('<h2 class="section-header">⚙️ Output Settings</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        column_name_option = st.selectbox(
            "Prediction column name",
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
        label_option = st.selectbox(
            "Prediction labels",
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
                custom_leave_label = st.text_input("Label for LEAVING", value="Leaving", key="custom_leave_label")
            with c2:
                custom_stay_label = st.text_input("Label for STAYING", value="Staying", key="custom_stay_label")
            prediction_labels = {
                1: custom_leave_label.strip() or "Leaving",
                0: custom_stay_label.strip() or "Staying",
            }
        else:
            prediction_labels = label_mappings[label_option]

        st.info(f"📌 Label Preview: Leaving → '{prediction_labels[1]}' | Staying → '{prediction_labels[0]}'")

    st.markdown("---")
    st.markdown('<h2 class="section-header">📊 Extra Options</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        include_probabilities = st.checkbox(
            "Include prediction probabilities in output",
            value=True,
            key="include_probabilities",
        )

    with col2:
        include_high_risk_download = st.checkbox(
            "Enable high-risk employees download (>50% leave prob)",
            value=True,
            key="include_high_risk",
        )

    if uploaded_file is not None:
        st.markdown("---")
        st.markdown('<h2 class="section-header">📄 Uploaded Data Preview</h2>', unsafe_allow_html=True)

        try:
            if file_format == "CSV":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Stats
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f"""<div class="stats-card"><h4>Total Rows</h4><div class="number">{len(df):,}</div></div>""",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""<div class="stats-card"><h4>Total Columns</h4><div class="number">{len(df.columns):,}</div></div>""",
                    unsafe_allow_html=True,
                )
            with c3:
                available_features = [col for col in BEST_FEATURES if col in df.columns]
                st.markdown(
                    f"""<div class="stats-card"><h4>Required Cols Found</h4><div class="number">{len(available_features)}/{len(BEST_FEATURES)}</div></div>""",
                    unsafe_allow_html=True,
                )

            st.dataframe(df.head(12), use_container_width=True)

            missing_columns = [col for col in BEST_FEATURES if col not in df.columns]
            if missing_columns:
                st.error(f"❌ Missing required columns: {missing_columns}")
            else:
                with st.expander("🔍 View columns being used for prediction"):
                    for feature in BEST_FEATURES:
                        st.write(f"• **{feature}** — sample: {df[feature].head(3).tolist()}")

                st.markdown("---")
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    batch_predict_button = st.button("🔮 Generate Batch Predictions", use_container_width=True, key="batch_predict")

                if batch_predict_button:
                    with st.spinner("🔄 Running model..."):
                        input_features = df[BEST_FEATURES].copy()
                        predictions = model.predict(input_features)
                        prediction_probabilities = model.predict_proba(input_features)

                        result_df = df.copy()
                        result_df[prediction_column_name] = [prediction_labels[int(p)] for p in predictions]

                        if include_probabilities:
                            result_df[f"{prediction_column_name}_Probability_Stay"] = (prediction_probabilities[:, 0] * 100).round(2)
                            result_df[f"{prediction_column_name}_Probability_Leave"] = (prediction_probabilities[:, 1] * 100).round(2)

                    fire_confetti()
                    st.success("✅ Predictions generated successfully!")

                    st.markdown("---")
                    st.markdown('<h2 class="section-header">📈 Results Summary</h2>', unsafe_allow_html=True)

                    leaving_count = int((predictions == 1).sum())
                    staying_count = int((predictions == 0).sum())
                    leaving_percentage = (leaving_count / len(predictions)) * 100
                    staying_percentage = (staying_count / len(predictions)) * 100
                    avg_leave_prob = float(prediction_probabilities[:, 1].mean() * 100)

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown(
                            f"""<div class="stats-card"><h4>Total Employees</h4><div class="number">{len(predictions):,}</div></div>""",
                            unsafe_allow_html=True,
                        )
                    with c2:
                        st.markdown(
                            f"""<div class="stats-card"><h4>Predicted to Leave</h4><div class="number" style="color: rgba(239,68,68,0.95);">{leaving_count:,}</div></div>""",
                            unsafe_allow_html=True,
                        )
                    with c3:
                        st.markdown(
                            f"""<div class="stats-card"><h4>Predicted to Stay</h4><div class="number" style="color: rgba(34,197,94,0.95);">{staying_count:,}</div></div>""",
                            unsafe_allow_html=True,
                        )
                    with c4:
                        st.markdown(
                            f"""<div class="stats-card"><h4>Avg. Leave Probability</h4><div class="number" style="color: rgba(245,158,11,0.95);">{avg_leave_prob:.1f}%</div></div>""",
                            unsafe_allow_html=True,
                        )

                    st.markdown("#### Turnover Distribution")
                    l, r = st.columns(2)
                    with l:
                        st.markdown(f"**Staying:** {staying_percentage:.1f}%")
                        st.markdown(
                            f"""
                            <div class="progress-wrap">
                              <div class="progress-fill progress-good" style="width:{staying_percentage:.2f}%;"></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with r:
                        st.markdown(f"**Leaving:** {leaving_percentage:.1f}%")
                        st.markdown(
                            f"""
                            <div class="progress-wrap">
                              <div class="progress-fill progress-bad" style="width:{leaving_percentage:.2f}%;"></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("---")
                    st.markdown('<h2 class="section-header">📄 Result Preview</h2>', unsafe_allow_html=True)
                    st.dataframe(result_df.head(25), use_container_width=True)

                    st.markdown("---")
                    st.markdown('<h2 class="section-header">📥 Download</h2>', unsafe_allow_html=True)

                    if include_high_risk_download:
                        d1, d2, d3 = st.columns(3)
                    else:
                        d1, d2 = st.columns(2)

                    with d1:
                        st.download_button(
                            "📥 Download as CSV",
                            data=convert_df_to_csv(result_df),
                            file_name="employee_predictions.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="download_csv",
                        )
                    with d2:
                        st.download_button(
                            "📥 Download as Excel",
                            data=convert_df_to_excel(result_df),
                            file_name="employee_predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key="download_excel",
                        )

                    if include_high_risk_download:
                        with d3:
                            high_risk_mask = prediction_probabilities[:, 1] > 0.5
                            high_risk_df = result_df.loc[high_risk_mask].copy()
                            if len(high_risk_df) > 0:
                                st.download_button(
                                    f"📥 High Risk Only ({len(high_risk_df):,})",
                                    data=convert_df_to_csv(high_risk_df),
                                    file_name="high_risk_employees.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                    key="download_high_risk",
                                )
                            else:
                                st.info("No high-risk employees found (leave probability > 50%).")

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please ensure your file is properly formatted and not corrupted.")
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

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.markdown(
        f"""
        <div class="hero-wrap">
          <div class="hero-inner">
            <div class="hero-title">👥 Employee Turnover Prediction</div>
            <div class="hero-sub">A neon-glass experience • Predict whether employees are likely to leave the company</div>
            <div class="badges">
              <div class="badge">Model: <strong>Random Forest</strong></div>
              <div class="badge">Inputs: <strong>{len(BEST_FEATURES)} Key Signals</strong></div>
              <div class="badge">Repo: <strong>{HF_REPO_ID}</strong></div>
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

if __name__ == "__main__":
    main()
