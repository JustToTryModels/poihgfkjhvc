import streamlit as st
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import io
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="FuturePulse | Employee Turnover AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# EXTREME CUSTOM CSS (JAW-DROPPING UI)
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

    /* Global Overrides */
    * { font-family: 'Poppins', sans-serif; }
    
    .main {
        background: radial-gradient(circle at top right, #0a192f, #112240, #020c1b);
        color: #e6f1ff;
    }

    /* Animated Background Blobs */
    .blob {
        position: fixed;
        width: 500px;
        height: 500px;
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.1) 0%, rgba(255, 0, 128, 0.1) 100%);
        filter: blur(80px);
        border-radius: 50%;
        z-index: -1;
        animation: float 20s infinite alternate;
    }
    @keyframes float {
        0% { transform: translate(-10%, -10%) rotate(0deg); }
        100% { transform: translate(20%, 20%) rotate(360deg); }
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        transition: all 0.4s ease;
        margin-bottom: 20px;
    }
    .glass-card:hover {
        border: 1px solid rgba(0, 245, 255, 0.4);
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 245, 255, 0.1);
    }

    /* Headers */
    .main-header {
        background: linear-gradient(to right, #00f5ff, #7117ea, #ea4444);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        letter-spacing: -2px;
        margin-bottom: 0;
    }
    .sub-header {
        color: #8892b0;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 3rem;
        text-transform: uppercase;
        letter-spacing: 5px;
    }

    /* Custom Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 50px !important;
        padding: 10px 40px !important;
        height: 60px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #8892b0 !important;
        transition: all 0.3s !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00f5ff, #7117ea) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.5) !important;
    }

    /* Input Fields styling */
    .stNumberInput input, .stSlider > div {
        background-color: rgba(0,0,0,0.2) !important;
        color: white !important;
        border-radius: 10px !important;
    }

    /* ULTRA NEON BUTTON */
    .stButton>button {
        width: 100%;
        height: 70px;
        background: linear-gradient(45deg, #FF0080, #7928CA, #0070F3, #00DFD8);
        background-size: 300% 300%;
        border: none !important;
        border-radius: 15px !important;
        color: white !important;
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 4px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        animation: gradientShift 4s ease infinite, pulse 2s infinite;
        transition: all 0.4s ease;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 245, 255, 0.4); }
        70% { box-shadow: 0 0 0 20px rgba(0, 245, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 245, 255, 0); }
    }
    .stButton>button:hover {
        transform: scale(1.02) translateY(-3px);
        filter: brightness(1.2);
    }

    /* Prediction Box */
    .result-card {
        padding: 3rem;
        border-radius: 30px;
        text-align: center;
        border: 2px solid;
        margin-top: 2rem;
        animation: slideIn 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stay-box {
        background: rgba(40, 167, 69, 0.1);
        border-color: #28a745;
        box-shadow: 0 0 30px rgba(40, 167, 69, 0.2);
    }
    .leave-box {
        background: rgba(220, 53, 69, 0.1);
        border-color: #dc3545;
        box-shadow: 0 0 30px rgba(220, 53, 69, 0.2);
    }

    /* Metrics and Progress */
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .custom-progress-bg {
        background: rgba(255,255,255,0.1);
        height: 20px;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    .custom-progress-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-in-out;
    }

    /* Hide Streamlit Footer */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
<div class="blob"></div>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION & MODEL LOADING
# ============================================================================
HF_REPO_ID = "IamPradeep/Employee-Churn-Predictor"
MODEL_FILENAME = "final_random_forest_model.joblib"
BEST_FEATURES = ["satisfaction_level", "time_spend_company", "average_monthly_hours", "number_project", "last_evaluation"]

@st.cache_resource
def load_model_from_huggingface():
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Model Load Failure: {e}")
        return None

# ============================================================================
# CALLBACKS FOR INPUT SYNC
# ============================================================================
def sync_val(src_key, dst_key):
    st.session_state[dst_key] = st.session_state[src_key]

# ============================================================================
# UI COMPONENTS
# ============================================================================
def display_prediction_ui(prob_leave, prediction):
    prob_stay = 100 - prob_leave
    status_class = "stay-box" if prediction == 0 else "leave-box"
    status_text = "STAYING" if prediction == 0 else "LEAVING"
    status_icon = "✅" if prediction == 0 else "⚠️"
    
    st.markdown(f"""
    <div class="result-card {status_class}">
        <h1 style="font-size: 4rem; margin:0;">{status_icon} {status_text}</h1>
        <p style="font-size: 1.5rem; opacity: 0.8;">The AI Predicts this employee will {status_text.lower()}</p>
        <div style="display: flex; justify-content: space-around; margin-top: 2rem;">
            <div>
                <p style="color: #28a745; font-weight: bold;">Stability Chance</p>
                <p class="metric-value" style="color: #28a745;">{prob_stay:.1f}%</p>
            </div>
            <div style="width: 2px; background: rgba(255,255,255,0.1);"></div>
            <div>
                <p style="color: #dc3545; font-weight: bold;">Churn Risk</p>
                <p class="metric-value" style="color: #dc3545;">{prob_leave:.1f}%</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    model = load_model_from_huggingface()
    
    st.markdown('<h1 class="main-header">FUTUREPULSE AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Employee Retention Analytics</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🎯 SINGLE ANALYSIS", "📂 BATCH ENGINE"])

    with tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        # Initialize session state
        if 'sat_val' not in st.session_state: st.session_state.sat_val = 0.5
        if 'eval_val' not in st.session_state: st.session_state.eval_val = 0.5

        with col1:
            st.write("### 😊 Satisfaction Level")
            st.slider("Sat Slider", 0.0, 1.0, key="sat_s", on_change=sync_val, args=("sat_s", "sat_val"), label_visibility="collapsed")
            st.number_input("Sat Num", 0.0, 1.0, key="sat_val", step=0.01, label_visibility="collapsed")
        
        with col2:
            st.write("### 📊 Performance Score")
            st.slider("Eval Slider", 0.0, 1.0, key="eval_s", on_change=sync_val, args=("eval_s", "eval_val"), label_visibility="collapsed")
            st.number_input("Eval Num", 0.0, 1.0, key="eval_val", step=0.01, label_visibility="collapsed")
        
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            years = st.number_input("📅 Years at Company", 1, 30, 3)
        with c2:
            projects = st.number_input("📁 Projects Handled", 1, 15, 4)
        with c3:
            hours = st.number_input("⏰ Monthly Hours", 50, 400, 200)
        
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("✨ RUN PREDICTIVE ANALYSIS"):
            with st.spinner("Decoding employee behavioral patterns..."):
                time.sleep(1)
                input_df = pd.DataFrame([[st.session_state.sat_val, years, hours, projects, st.session_state.eval_val]], columns=BEST_FEATURES)
                pred = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0][1] * 100
                display_prediction_ui(prob, pred)

    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("### 📤 Upload Workforce Data")
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.write("#### Data Preview")
            st.dataframe(df.head(5), use_container_width=True)
            
            missing = [c for c in BEST_FEATURES if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                if st.button("🚀 PROCESS BATCH PREDICTIONS"):
                    results = model.predict(df[BEST_FEATURES])
                    probs = model.predict_proba(df[BEST_FEATURES])[:, 1]
                    
                    df['AI_Prediction'] = ["Leave" if r == 1 else "Stay" for r in results]
                    df['Risk_Score (%)'] = (probs * 100).round(2)
                    
                    st.success("Analysis Complete!")
                    st.dataframe(df, use_container_width=True)
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Enhanced Report", csv, "workforce_analysis.csv", "text/csv")
        else:
            st.info("Please upload a file containing: satisfaction_level, time_spend_company, average_monthly_hours, number_project, last_evaluation")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
