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
    page_title="AI Employee Retention",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# JAW-DROPPING CSS STYLING (DARK MODE & NEON AESTHETIC)
# ============================================================================
st.markdown("""
<style>
    /* === GLOBAL RESET & FONTS === */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    .stApp {
        background-color: #050505;
        color: #ffffff;
        font-family: 'Outfit', sans-serif;
    }

    /* === ANIMATED BACKGROUND MESH === */
    body {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a0b2e);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        margin: 0;
        overflow-x: hidden;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* === CONTAINER STYLING === */
    .block-container {
        max-width: 1300px !important;
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* === HEADERS === */
    .main-header {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2ff, #ff0055);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
        text-shadow: 0 10px 30px rgba(0, 242, 255, 0.3);
        animation: fadeInDown 1s ease-out;
    }
    
    .sub-header {
        font-size: 1.4rem;
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
        animation: fadeInUp 1s ease-out 0.2s backwards;
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* === GLASSMORPHISM CARDS === */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        padding: 2rem;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px 0 rgba(0, 242, 255, 0.15);
        border-color: rgba(0, 242, 255, 0.3);
    }

    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #00f2ff;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* === CUSTOM SLIDERS (Hack for Streamlit) === */
    /* This is tricky because Streamlit injects styles, but we can target specific classes */
    .stSlider > div[data-testid="stSlider"] > div > div > div {
        background: linear-gradient(90deg, #00f2ff, #0066ff);
        border-radius: 10px;
    }
    
    input[type=range] {
        -webkit-appearance: none; 
        background: transparent; 
    }
    
    input[type=range]::-webkit-slider-thumb {
        -webkit-appearance: none;
        height: 20px;
        width: 20px;
        border-radius: 50%;
        background: #ffffff;
        cursor: pointer;
        margin-top: -8px; 
        box-shadow: 0 0 10px rgba(255,255,255,0.8);
    }

    input[type=range]::-webkit-slider-runnable-track {
        width: 100%;
        height: 4px;
        cursor: pointer;
        background: rgba(255,255,255,0.2);
        border-radius: 2px;
    }

    /* === NUMBER INPUTS === */
    .stNumberInput > div > div > input {
        background-color: rgba(255,255,255,0.1) !important;
        color: #fff !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 8px;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #00f2ff !important;
        box-shadow: 0 0 10px rgba(0, 242, 255, 0.2);
    }

    /* === TABS - MASSIVE & BOLD === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        justify-content: center;
        padding: 2rem 0;
        background: rgba(255,255,255,0.02);
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 100px !important;
        min-width: 300px !important;
        background: rgba(0,0,0,0.3);
        border-radius: 15px !important;
        font-weight: 800 !important;
        font-size: 1.5rem !important;
        color: #aaa !important;
        border: 2px solid transparent !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stTabs [data-baseweb="tab"] p {
        font-weight: 800 !important;
        font-size: 1.5rem !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 242, 255, 0.1), rgba(255, 0, 85, 0.1)) !important;
        color: #fff !important;
        border: 2px solid rgba(0, 242, 255, 0.5) !important;
        box-shadow: 0 0 30px rgba(0, 242, 255, 0.2) !important;
        text-shadow: 0 0 10px rgba(0, 242, 255, 0.5);
        transform: scale(1.05);
    }

    /* === ULTRA PREDICT BUTTON === */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #00f2ff, #0066ff) !important;
        color: #000 !important;
        font-size: 1.5rem;
        font-weight: 900 !important;
        padding: 1.2rem;
        border-radius: 50px;
        border: none !important;
        transition: all 0.3s ease;
        position: relative;
        z-index: 1;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 10px 30px rgba(0, 242, 255, 0.4);
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #ff0055, #ff00cc) !important;
        color: #fff !important;
        box-shadow: 0 10px 40px rgba(255, 0, 85, 0.5);
        transform: translateY(-3px) scale(1.01);
    }

    .stButton > button:active {
        transform: translateY(1px);
    }

    /* === RESULT BOXES === */
    .result-box {
        padding: 2.5rem;
        border-radius: 25px;
        text-align: center;
        margin-top: 2rem;
        animation: popIn 0.6s cubic-bezier(0.68, -0.55, 0.27, 1.55);
    }
    
    @keyframes popIn {
        0% { opacity: 0; transform: scale(0.8); }
        100% { opacity: 1; transform: scale(1); }
    }

    .stay-result {
        background: rgba(0, 255, 136, 0.1);
        border: 2px solid #00ff88;
        box-shadow: 0 0 50px rgba(0, 255, 136, 0.2);
    }
    
    .stay-result h1 { color: #00ff88; text-shadow: 0 0 20px rgba(0, 255, 136, 0.5); }

    .leave-result {
        background: rgba(255, 0, 85, 0.1);
        border: 2px solid #ff0055;
        box-shadow: 0 0 50px rgba(255, 0, 85, 0.2);
    }

    .leave-result h1 { color: #ff0055; text-shadow: 0 0 20px rgba(255, 0, 85, 0.5); }

    /* === PROGRESS BARS (NEON) === */
    .neon-progress-container {
        width: 100%;
        background: rgba(255,255,255,0.1);
        height: 20px;
        border-radius: 10px;
        overflow: hidden;
        margin-top: 10px;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.5);
    }

    .neon-bar-green {
        height: 100%;
        background: linear-gradient(90deg, #00ff88, #00cc6a);
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
        transition: width 1s ease-in-out;
    }

    .neon-bar-red {
        height: 100%;
        background: linear-gradient(90deg, #ff0055, #ff0088);
        box-shadow: 0 0 20px rgba(255, 0, 85, 0.5);
        transition: width 1s ease-in-out;
    }

    /* === DATAFRAME STYLING === */
    .stDataFrame {
        background-color: rgba(255,255,255,0.05);
        border-radius: 10px;
    }

    /* === EXPANDER === */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        color: #00f2ff;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
HF_REPO_ID = "IamPradeep/Employee-Churn-Predictor"
MODEL_FILENAME = "final_random_forest_model.joblib"

# Define the 5 selected features (in exact order)
BEST_FEATURES = [
    "satisfaction_level",
    "time_spend_company", 
    "average_monthly_hours",
    "number_project",
    "last_evaluation"
]

FEATURE_DESCRIPTIONS = {
    "satisfaction_level": "Employee satisfaction level (0.0 - 1.0)",
    "time_spend_company": "Years at company (integer)",
    "average_monthly_hours": "Average monthly hours worked (integer)",
    "number_project": "Number of projects (integer)",
    "last_evaluation": "Last performance evaluation score (0.0 - 1.0)"
}

# ============================================================================
# LOAD MODEL FROM HUGGING FACE
# ============================================================================
@st.cache_resource
def load_model_from_huggingface():
    """Load the trained model from Hugging Face Hub"""
    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model"
        )
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# ============================================================================
# CALLBACK FUNCTIONS FOR SYNCING
# ============================================================================
def sync_satisfaction_slider():
    st.session_state.satisfaction_level = st.session_state.sat_slider

def sync_satisfaction_input():
    st.session_state.satisfaction_level = st.session_state.sat_input

def sync_evaluation_slider():
    st.session_state.last_evaluation = st.session_state.eval_slider

def sync_evaluation_input():
    st.session_state.last_evaluation = st.session_state.eval_input

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
    return output.getvalue()

# ============================================================================
# INDIVIDUAL PREDICTION TAB
# ============================================================================
def render_individual_prediction_tab(model):
    st.markdown('<div class="glass-card"><h2 style="text-align:center; color:#fff;">🔮 Input Parameters</h2></div>', unsafe_allow_html=True)

    # Initialize state
    if 'satisfaction_level' not in st.session_state:
        st.session_state.satisfaction_level = 0.50
    if 'last_evaluation' not in st.session_state:
        st.session_state.last_evaluation = 0.70

    # ROW 1: Satisfaction & Evaluation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="feature-title">😊 Satisfaction Level</div>
        </div>
        """, unsafe_allow_html=True)
        
        s_col1, s_col2 = st.columns([4, 1])
        with s_col1:
            st.slider("", 0.0, 1.0, st.session_state.satisfaction_level, 0.01, 
                      label_visibility="collapsed", key="sat_slider", on_change=sync_satisfaction_slider)
        with s_col2:
            st.number_input("", 0.0, 1.0, st.session_state.satisfaction_level, 0.01, format="%.2f",
                           label_visibility="collapsed", key="sat_input", on_change=sync_satisfaction_input)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <div class="feature-title">📊 Last Evaluation</div>
        </div>
        """, unsafe_allow_html=True)
        
        e_col1, e_col2 = st.columns([4, 1])
        with e_col1:
            st.slider("", 0.0, 1.0, st.session_state.last_evaluation, 0.01, 
                      label_visibility="collapsed", key="eval_slider", on_change=sync_evaluation_slider)
        with e_col2:
            st.number_input("", 0.0, 1.0, st.session_state.last_evaluation, 0.01, format="%.2f",
                           label_visibility="collapsed", key="eval_input", on_change=sync_evaluation_input)

    # ROW 2: Years, Projects, Hours
    col3, col4, col5 = st.columns(3)

    with col3:
        st.markdown("""
        <div class="glass-card">
            <div class="feature-title">📅 Years at Company</div>
        </div>
        """, unsafe_allow_html=True)
        time_spend_company = st.number_input("Years", 1, 40, 3, 1, label_visibility="collapsed", key="ind_years")

    with col4:
        st.markdown("""
        <div class="glass-card">
            <div class="feature-title">📁 Number of Projects</div>
        </div>
        """, unsafe_allow_html=True)
        number_project = st.number_input("Projects", 1, 10, 4, 1, label_visibility="collapsed", key="ind_projects")

    with col5:
        st.markdown("""
        <div class="glass-card">
            <div class="feature-title">⏰ Avg. Monthly Hours</div>
        </div>
        """, unsafe_allow_html=True)
        average_monthly_hours = st.number_input("Hours", 80, 350, 200, 5, label_visibility="collapsed", key="ind_hours")

    input_data = {
        'satisfaction_level': st.session_state.satisfaction_level,
        'time_spend_company': time_spend_company,
        'average_monthly_hours': average_monthly_hours,
        'number_project': number_project,
        'last_evaluation': st.session_state.last_evaluation
    }

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_a, col_b, col_c = st.columns([1, 3, 1])
    with col_b:
        predict_clicked = st.button("🔮 ANALYZE EMPLOYEE", use_container_width=True, key="btn_predict")

    if predict_clicked:
        input_df = pd.DataFrame([input_data])[BEST_FEATURES]
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        prob_stay = prediction_proba[0] * 100
        prob_leave = prediction_proba[1] * 100
        
        st.markdown("---")
        
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            if prediction == 0:
                st.markdown(f"""
                <div class="result-box stay-result">
                    <h1 style="font-size: 3rem;">✅ RETAINED</h1>
                    <p style="font-size: 1.5rem; color: #fff;">The employee is predicted to <strong>STAY</strong>.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box leave-result">
                    <h1 style="font-size: 3rem;">⚠️ ATTRITION</h1>
                    <p style="font-size: 1.5rem; color: #fff;">The employee is predicted to <strong>LEAVE</strong>.</p>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="glass-card"><h3 style="text-align:center; color:#00f2ff;">📊 Probability Analysis</h3></div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="margin-top: 20px;">
                <div style="display:flex; justify-content:space-between; font-weight:bold;">
                    <span>STAY Probability</span>
                    <span style="color:#00ff88">{prob_stay:.1f}%</span>
                </div>
                <div class="neon-progress-container">
                    <div class="neon-bar-green" style="width: {prob_stay}%;"></div>
                </div>
            </div>
            
            <div style="margin-top: 20px;">
                <div style="display:flex; justify-content:space-between; font-weight:bold;">
                    <span>LEAVE Probability</span>
                    <span style="color:#ff0055">{prob_leave:.1f}%</span>
                </div>
                <div class="neon-progress-container">
                    <div class="neon-bar-red" style="width: {prob_leave}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# BATCH PREDICTION TAB
# ============================================================================
def render_batch_prediction_tab(model):
    st.markdown('<div class="glass-card"><h2 style="text-align:center; color:#fff;">📊 Bulk Data Processing</h2></div>', unsafe_allow_html=True)

    with st.expander("📋 Required Columns (Click to Expand)", expanded=False):
        st.markdown("""
        <div style="background: rgba(0, 242, 255, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(0, 242, 255, 0.3);">
            Your file must strictly contain these columns:
        </div>
        """, unsafe_allow_html=True)
        cols = st.columns(3)
        for i, feat in enumerate(BEST_FEATURES):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="padding: 10px; margin-top: 10px; border-left: 3px solid #ff0055; background: rgba(255,255,255,0.05);">
                    <strong style="color: #fff;">{feat}</strong><br/>
                    <small style="color: #aaa;">{FEATURE_DESCRIPTIONS[feat]}</small>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Upload Area
    col1, col2 = st.columns([1, 2])
    with col1:
        file_format = st.selectbox("📁 File Type", ["CSV", "Excel (.xlsx)"], label_visibility="visible")
    
    with col2:
        if file_format == "CSV":
            uploaded_file = st.file_uploader("Upload Dataset", type=["csv"], label_visibility="collapsed")
        else:
            uploaded_file = st.file_uploader("Upload Dataset", type=["xlsx", "xls"], label_visibility="collapsed")

    # Settings
    c1, c2 = st.columns(2)
    with c1:
        col_opt = st.selectbox("Prediction Column Name", ["Prediction", "Turnover", "Churn_Risk"])
    with c2:
        include_prob = st.checkbox("Include Probability Columns", value=True)

    st.markdown("---")

    if uploaded_file is not None:
        try:
            if file_format == "CSV":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Validation
            missing = [col for col in BEST_FEATURES if col not in df.columns]
            
            if missing:
                st.error(f"❌ Missing columns: {', '.join(missing)}")
            else:
                st.success(f"✅ File loaded successfully! Rows: {len(df)}")
                
                # Predict Button
                if st.button("🚀 RUN BATCH PREDICTION", use_container_width=True, type="primary"):
                    with st.spinner("AI Model Processing..."):
                        input_features = df[BEST_FEATURES].copy()
                        predictions = model.predict(input_features)
                        probs = model.predict_proba(input_features)
                        
                        result_df = df.copy()
                        result_df[col_opt] = ["Leave" if p == 1 else "Stay" for p in predictions]
                        
                        if include_prob:
                            result_df[f"{col_opt}_Prob_Stay"] = (probs[:, 0] * 100).round(2)
                            result_df[f"{col_opt}_Prob_Leave"] = (probs[:, 1] * 100).round(2)
                    
                    st.success("Processing Complete!")
                    
                    # Stats
                    st.markdown('<div class="glass-card"><h3 style="color:#00f2ff;">📈 Analysis Summary</h3></div>', unsafe_allow_html=True)
                    
                    leave_cnt = sum(predictions == 1)
                    stay_cnt = sum(predictions == 0)
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Employees", len(df))
                    m2.metric("High Risk (Leaving)", f"{leave_cnt} ({leave_cnt/len(df)*100:.1f}%)")
                    m3.metric("Stable (Staying)", f"{stay_cnt} ({stay_cnt/len(df)*100:.1f}%)")

                    st.dataframe(result_df.head(10), use_container_width=True)
                    
                    # Download
                    d1, d2 = st.columns(2)
                    csv = convert_df_to_csv(result_df)
                    d1.download_button("📥 Download CSV", csv, "predictions.csv", "text/csv")
                    
                    high_risk = result_df[probs[:, 1] > 0.5]
                    if len(high_risk) > 0:
                        hr_csv = convert_df_to_csv(high_risk)
                        d2.download_button(f"⚠️ Download High Risk ({len(high_risk)})", hr_csv, "high_risk.csv", "text/csv")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("👆 Upload a file to begin batch processing.")

# ============================================================================
# MAIN
# ============================================================================
def main():
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div class="main-header">AI WORKFORCE GUARDIAN</div>
        <div class="sub-header">Advanced Employee Turnover Prediction Engine</div>
    </div>
    """, unsafe_allow_html=True)
    
    model = load_model_from_huggingface()
    if not model:
        st.stop()

    # Tabs
    tab1, tab2 = st.tabs(["🔮 Single Employee Analysis", "📊 Batch Processing"])
    
    with tab1:
        render_individual_prediction_tab(model)
    
    with tab2:
        render_batch_prediction_tab(model)

if __name__ == "__main__":
    main()
