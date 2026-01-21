import streamlit as st
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import io

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🔮 NEON TURNOVER ORACLE",
    page_icon="💫",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# INSANELY BEAUTIFUL CUSTOM CSS – THIS IS WHERE THE MAGIC HAPPENS
# ============================================================================
st.markdown("""
<style>
    /* Full-screen animated cosmic background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(-45deg, #12081a, #1a0b2e, #2e0854, #4a00e0, #8e2de2, #ff006e, #ff4757);
        background-size: 600% 600%;
        animation: cosmicFlow 25s ease infinite;
    }
    
    @keyframes cosmicFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Frosted glass main container */
    .block-container {
        background: rgba(15, 5, 30, 0.65) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-radius: 30px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.6),
            0 0 100px rgba(142, 45, 226, 0.4),
            inset 0 0 80px rgba(255, 255, 255, 0.1) !important;
        max-width: 1400px !important;
        padding: 3rem 4rem !important;
        margin: 2rem auto !important;
    }

    /* Rainbow neon title – this alone will make people gasp */
    .main-header {
        font-size: 4.8rem !important;
        font-weight: 900 !important;
        text-align: center;
        background: linear-gradient(90deg, #ff0000, #ff9100, #ffea00, #00ff73, #00ffff, #8b00ff, #ff00c8, #ff0000);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        backgroundikera-size: 400% 400%;
        animation: rainbowFlow 8s linear infinite, glowPulse 3s ease-in-out infinite;
        text-shadow: 0 0 40px rgba(255, 255, 255, 0.6), 0 0 80px rgba(142, 45, 226, 0.8);
        letter-spacing: 4px;
        margin-bottom: 0.5rem !important;
    }
    
    @keyframes rainbowFlow {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    
    @keyframes glowPulse {
        0%, 100% { text-shadow: 0 0 40px rgba(255, 255, 255, 0.6); }
        50% { text-shadow: 0 0 80px rgba(255, 0, 255, 0.9), 0 0 120px rgba(142, 45, 226, 1); }
    }

    .sub-header {
        font-size: 1.6rem;
        color: #e0aaff !important;
        text-align: center;
        font-weight: 600;
        text-shadow: 0 0 20px rgba(224, 170, 255, 0.6);
        margin-bottom: 3rem;
    }

    /* Ultra tabs – floating neon pills */
    .stTabs [data-baseweb="tab-list"] {
        gap: 25px;
        justify-content: center;
        background: transparent;
        padding: 20px 0;
        border-bottom: 2px solid rgba(142, 45, 226, 0.5);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 100px !important;
        min-width: 380px !important;
        padding: 0 60px !important;
        background: rgba(15, 5, 30, 0.7) !important;
        border-radius: 25px 25px 0 0 !important;
        font-weight: 900 !important;
        font-size: 1.6rem !important;
        color: #e0aaff !important;
        border: 2px solid rgba(142, 45, 226, 0.6) !important;
        backdrop-filter: blur(10px);
        transition: all 0.4s ease !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-10px) scale(1.05);
        background: rgba(142, 45, 226, 0.4) !important;
        box-shadow: 0 20px 50px rgba(142, 45, 226, 0.6) !important;
        color: white !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8e2de2, #4a00e0) !important;
        color: white !important;
        box-shadow: 0 20px 60px rgba(142, 45, 226, 0.8) !important;
        border-bottom: none !important;
        transform: translateY(-5px);
    }

    /* Feature cards – glowing glass */
    .feature-card {
        background: rgba(255, 255, 255, 0.12) !important;
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 
            0 15px 35px rgba(0, 0, 0, 0.4),
            0 0 40px rgba(142, 45, 226, 0.4);
        transition: all 0.4s ease;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #e0aaff;
    }
    
    .feature-card:hover {
        transform: translateY(-15px) scale(1.05);
        box-shadow: 0 25px 60px rgba(142, 45, 226, 0.7) !important;
    }

    /* Prediction result boxes – PURE DRAMA */
    .prediction-box {
        padding: 3rem;
        border-radius: 25px;
        text-align: center;
        margin: 2rem 0;
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-20px); }
    }
    
    .stay-prediction {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.3), rgba(0, 255, 128, 0.2));
        border: 3px solid #00ff88;
        box-shadow: 
            0 0 60px rgba(0, 255, 136, 0.8),
            0 0 120px rgba(0, 255, 136, 0.5);
        animation: glowGreen 2s ease-in-out infinite alternate;
    }
    
    .leave-prediction {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.3), rgba(255, 0, 128, 0.2));
        border: 3px solid #ff006e;
        box-shadow: 
            0 0 60px rgba(255, 0, 110, 0.9),
            0 0 120px rgba(255, 0, 110, 0.6);
        animation: glowRed 2s ease-in-out infinite alternate;
    }
    
    @keyframes glowGreen {
        from { box-shadow: 0 0 60px rgba(0, 255, 136, 0.8); }
        to { box-shadow: 0 0 100px rgba(0, 255, 136, 1), 0 0 150px rgba(0, 255, 136, 0.8); }
    }
    
    @keyframes glowRed {
        from { box-shadow: 0 0 60px rgba(255, 0, 110, 0.9); }
        to { box-shadow: 0 0 100px rgba(255, 0, 110, 1), 0 0 150px rgba(255, 0, 110, 0.8); }
    }
    
    .prediction-box h1 {
        font-size: 6.5rem !important;
        font-weight: 900;
        margin: 0;
        text-shadow: 0 0 30px currentColor;
    }

    /* Ultimate predict button – THIS IS GOD-TIER */
    .stButton > button {
        width: 100%;
        height: 90px;
        font-size: 2rem !important;
        font-weight: 900 !important;
        letter-spacing: 4px;
        text-transform: uppercase;
        background: linear-gradient(45deg, #ff006e, #833ab4, #fd1d1d, #fcb045, #00f5ff, #ff006e);
        background-size: 600% 600%;
        color: white !important;
        border: none !important;
        border-radius: 50px;
        animation: insaneGradient 4s ease infinite, megaPulse 2s ease-in-out infinite;
        box-shadow: 
            0 0 50px rgba(255, 0, 110, 0.8),
            0 0 100px rgba(142, 45, 226, 0.6),
            0 20px 40px rgba(0, 0, 0, 0.5);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    @keyframes insaneGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes megaPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .stButton > button:hover {
        transform: translateY(-10px) scale(1.1);
        box-shadow: 0 0 120px rgba(255, 0, 110, 1), 0 0 200px rgba(142, 45, 226, 0.9) !important;
    }
    
    .stButton > button:active {
        transform:scale(0.95);
    }

    /* Progress bars with glow */
    .progress-bar-container {
        height: 30px;
        background: rgba(0,0,0,0.4);
        border-radius: 15px;
        overflow: hidden;
        border: 2px solid rgba(255,255,255,0.2);
    }
    
    .progress-bar-green, .progress-bar-red {
        height: 100%;
        border-radius: 15px;
        animation: barGlow 3s ease-in-out infinite;
    }
    
    .progress-bar-green { background: linear-gradient(90deg, #00ff88, #00f5ff); }
    .progress-bar-red { background: linear-gradient(90deg, #ff006e, #ff4757); }
    
    @keyframes barGlow {
        0%, 100% { box-shadow: 0 0 20px currentColor; }
        50% { box-shadow: 0 0 40px currentColor; }
    }

    /* Stats cards */
    .stats-card {
        background: rgba(15, 5, 30, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(142, 45, 226, 0.5);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    
    .stats-card .number {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #ff00c8, #00ffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIG & MODEL LOADING
# ============================================================================
HF_REPO_ID = "IamPradeep/Employee-Churn-Predictor"
MODEL_FILENAME = "final_random_forest_model.joblib"

BEST_FEATURES = [
    "satisfaction_level", "time_spend_company", "average_monthly_hours",
    "number_project", "last_evaluation"
]

FEATURE_DESCRIPTIONS = {
    "satisfaction_level": "Employee satisfaction level (0.0 - 1.0)",
    "time_spend_company": "Years at company",
    "average_monthly_hours": "Average monthly hours worked",
    "number_project": "Number of projects",
    "last_evaluation": "Last performance evaluation score (0.0 - 1.0)"
}

@st.cache_resource
def load_model():
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
        return joblib.load(path)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()
if not model:
    st.stop()

# ============================================================================
# SYNC CALLBACKS
# ============================================================================
def sync_sat_slider(): st.session_state.satisfaction_level = st.session_state.sat_slider
def sync_sat_input(): st.session_state.satisfaction_level = st.session_state.sat_input
def sync_eval_slider(): st.session_state.last_evaluation = st.session_state.eval_slider
def sync_eval_input(): st.session_state.last_evaluation = st.session_state.eval_input

# ============================================================================
# HELPERS
# ============================================================================
def csv_download(df): return df.to_csv(index=False).encode('utf-8')
def excel_download(df):
    output = io.BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    return output.getvalue()

# ============================================================================
# INDIVIDUAL PREDICTION
# ============================================================================
def individual_tab():
    st.markdown("### 💫 Enter Employee Details")
    
    if 'satisfaction_level' not in st.session_state: st.session_state.satisfaction_level = 0.5
    if 'last_evaluation' not in st.session_state: st.session_state.last_evaluation = 0.7
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-card">😊 Satisfaction Level</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([3,1])
        with c1: st.slider("", 0.0, 1.0, st.session_state.satisfaction_level, 0.01, key="sat_slider", on_change=sync_sat_slider)
        with c2: st.number_input("", 0.0, 1.0, st.session_state.satisfaction_level, 0.01, format="%.2f", key="sat_input", on_change=sync_sat_input, label_visibility="collapsed")
    
    with col2:
        st.markdown('<div class="feature-card">📊 Last Evaluation</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([3,1])
        with c1: st.slider("", 0.0, 1.0, st.session_state.last_evaluation, 0.01, key="eval_slider", on_change=sync_eval_slider, label_visibility="collapsed")
        with c2: st.number_input("", 0.0, 1.0, st.session_state.last_evaluation, 0.01, format="%.2f", key="eval_input", on_change=sync_eval_input, label_visibility="collapsed")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="feature-card">📅 Years at Company</div>', unsafe_allow_html=True)
        years = st.number_input("", 1, 40, 3, key="y1")
    with col2:
        st.markdown('<div class="feature-card">📁 Number of Projects</div>', unsafe_allow_html=True)
        projects = st.number_input("", 1, 10, 4, key="p1")
    with col3:
        st.markdown('<div class="feature-card">⏰ Avg Monthly Hours</div>', unsafe_allow_html=True)
        hours = st.number_input("", 80, 350, 200, 5, key="h1")
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 UNLEASH THE ORACLE", use_container_width=True):
        data = [[st.session_state.satisfaction_level, years, hours, projects, st.session_state.last_evaluation]]
        df = pd.DataFrame(data, columns=BEST_FEATURES)
        
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0]
        prob_stay = prob[0] * 100
        prob_leave = prob[1] * 100
        
        st.confetti()
        
        col1, col2 = st.columns(2)
        with col1:
            if pred == 0:
                st.markdown('<div class="prediction-box stay-prediction"><h1>STAY 💚</h1><p>They are staying!</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-box leave-prediction"><h1>LEAVE 🔥</h1><p>High risk of leaving!</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Confidence")
            st.write(f"**Stay:** {prob_stay:.1f}%")
            st.markdown(f'<div class="progress-bar-container"><div class="progress-bar-green" style="width:{prob_stay}%"></div></div>', unsafe_allow_html=True)
            st.write(f"**Leave:** {prob_leave:.1f}%")
            st.markdown(f'<div class="progress-bar-container"><div class="progress-bar-red" style="width:{prob_leave}%"></div></div>', unsafe_allow_html=True)

# ============================================================================
# BATCH PREDICTION
# ============================================================================
def batch_tab():
    st.markdown("### 📊 Batch Prediction Mode")
    
    with st.expander("📋 Required Columns", expanded=True):
        for f in BEST_FEATURES:
            st.markdown(f"**{f}** – {FEATURE_DESCRIPTIONS[f]}")
    
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    
    if uploaded:
        df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        
        missing = [col for col in BEST_FEATURES if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            pred_col = st.text_input("Prediction column name", "Turnover_Prediction")
        with col2:
            labels = st.selectbox("Labels", ["Stay/Leave", "No/Yes", "Safe/Risk", "0/1"])
            label_map = {"Stay/Leave": {0:"Stay",1:"Leave"}, "No/Yes":{0:"No",1:"Yes"}, "Safe/Risk":{0:"Safe",1:"Risk"}, "0/1":{0:"0",1:"1"}}
            labels_dict = label_map.get(labels, {0:"Stay",1:"Leave"})
        
        add_prob = st.checkbox("Add probability columns", True)
        
        if st.button("🚀 GENERATE PREDICTIONS", use_container_width=True):
            X = df[BEST_FEATURES]
            preds = model.predict(X)
            probs = model.predict_proba(X)
            
            result = df.copy()
            result[pred_col] = [labels_dict[p] for p in preds]
            if add_prob:
                result[f"{pred_col}_Stay_%"] = (probs[:,0]*100).round(1)
                result[f"{pred_col}_Leave_%"] = (probs[:,1]*100).round(1)
            
            st.balloons()
            st.success("Predictions Ready!")
            
            leave_count = sum(preds)
            st.markdown(f"### ⚡ {leave_count:,} employees predicted to leave ({leave_count/len(preds)*100:.1f}%)")
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Download Full Results (CSV)", csv_download(result), "turnover_predictions.csv", "text/csv")
            with col2:
                st.download_button("📥 Download Full Results (Excel)", excel_download(result), "turnover_predictions.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
            high_risk = result[probs[:,1] > 0.6]
            if len(high_risk) > 0:
                st.download_button(f"⚠️ Download HIGH-RISK Only ({len(high_risk)} employees)", csv_download(high_risk), "HIGH_RISK_EMPLOYEES.csv", "text/csv")

# ============================================================================
# MAIN APP
# ============================================================================
st.markdown('<h1 class="main-header">NEON TURNOVER ORACLE</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">The most beautiful employee churn predictor on Earth</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["✨ Individual Prediction", "🌌 Batch Prediction"])

with tab1:
    individual_tab()
with tab2:
    batch_tab()
