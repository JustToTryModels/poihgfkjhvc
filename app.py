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
    page_title="Employee Turnover Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# JAW-DROPPING & EYEBROW-RAISING UI STYLING
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.1/css/all.min.css');

    /* --- BASIC SETUP --- */
    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
    }

    /* --- DARK THEME & BACKGROUND --- */
    body {
        background-color: #0a192f; /* Dark Navy Blue */
    }
    .stApp {
        background-image: linear-gradient(180deg, rgba(10, 25, 47, 0.95) 0%, #0a192f 30%), url("https://www.transparenttextures.com/patterns/cubes.png");
        background-attachment: fixed;
    }
    
    /* --- CUSTOM FONT & TEXT COLORS --- */
    h1, h2, h3, h4, h5, h6, p, .stMarkdown, label, .st-cc, .st-bq {
        color: #ccd6f6; /* Light Slate */
    }
    
    /* --- MAIN CONTENT CONTAINER --- */
    .block-container {
        max-width: 1400px !important;
        padding: 2rem 4rem !important;
        animation: fadeIn 1s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* --- PAGE HEADERS --- */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        background: -webkit-linear-gradient(45deg, #64ffda, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { text-shadow: 0 0 10px #64ffda, 0 0 20px #64ffda, 0 0 30px #a855f7; }
        to { text-shadow: 0 0 20px #64ffda, 0 0 30px #a855f7, 0 0 40px #a855f7; }
    }
    .sub-header {
        font-size: 1.25rem;
        color: #8892b0; /* Slate */
        text-align: center;
        margin-bottom: 3rem;
    }

    /* --- GLASSMORPHISM CARDS --- */
    .glass-card {
        background: rgba(20, 39, 68, 0.5);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(100, 255, 218, 0.2);
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(100, 255, 218, 0.5);
        transform: translateY(-5px);
    }
    .glass-card h3 {
        color: #64ffda; /* Aqua */
        font-weight: 600;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #303c55;
        padding-bottom: 0.5rem;
    }

    /* --- TAB STYLING --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        justify-content: center;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px !important;
        background: transparent !important;
        border: 2px solid #303c55 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        color: #8892b0 !important;
        transition: all 0.3s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(100, 255, 218, 0.1) !important;
        color: #64ffda !important;
        border-color: #64ffda !important;
    }
    .stTabs [aria-selected="true"] {
        background: #64ffda !important;
        color: #0a192f !important;
        border-color: #64ffda !important;
        box-shadow: 0 0 15px #64ffda;
    }
    
    /* --- INPUT WIDGETS --- */
    .stSlider, .stNumberInput {
        background-color: rgba(10, 25, 47, 0.8);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: 1px solid #303c55;
    }
    /* Slider Track */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #64ffda !important;
    }
    /* Slider Handle */
    .stSlider [data-baseweb="slider"] div[role="slider"] > div {
        background-color: #64ffda !important;
        box-shadow: 0 0 10px #64ffda;
    }

    /* --- VIBRANT PREDICT BUTTON (Adapted from original) --- */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #f72585, #b5179e, #7209b7, #3a0ca3);
        background-size: 300% 300%;
        color: white !important;
        font-size: 1.4rem;
        font-weight: 700 !important;
        padding: 1rem 2rem;
        border-radius: 50px;
        border: none !important;
        box-shadow: 0 0 20px rgba(247, 37, 133, 0.6);
        transition: all 0.4s ease;
        animation: gradientShift 4s ease infinite;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 30px rgba(247, 37, 133, 1);
    }

    /* --- PREDICTION RESULT DISPLAY --- */
    .prediction-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 2rem;
    }
    .prediction-result {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .prediction-stay { color: #50fa7b; text-shadow: 0 0 15px #50fa7b; }
    .prediction-leave { color: #ff5555; text-shadow: 0 0 15px #ff5555; }
    .prediction-desc {
        font-size: 1.2rem;
        color: #ccd6f6;
    }

    /* --- GAUGE CHART FOR PROBABILITY --- */
    .gauge-container {
        position: relative;
        width: 200px;
        height: 100px;
        overflow: hidden;
    }
    .gauge-bg {
        width: 200px;
        height: 100px;
        border: 20px solid #303c55;
        border-bottom: none;
        border-radius: 100px 100px 0 0;
        box-sizing: border-box;
    }
    .gauge-fill {
        position: absolute;
        top: 0;
        left: 0;
        width: 200px;
        height: 100px;
        border: 20px solid;
        border-bottom: none;
        border-radius: 100px 100px 0 0;
        box-sizing: border-box;
        transform-origin: center bottom;
        transition: transform 1.5s cubic-bezier(0.19, 1, 0.22, 1);
    }
    .gauge-fill-leave { border-color: #ff5555; }
    .gauge-fill-stay { border-color: #50fa7b; }
    .gauge-cover {
        position: absolute;
        top: 20px;
        left: 20px;
        width: 160px;
        height: 80px;
        background: transparent;
        border-radius: 80px 80px 0 0;
    }
    .gauge-text {
        position: absolute;
        bottom: -15px;
        width: 100%;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
    }
    .gauge-label {
        position: absolute;
        bottom: -40px;
        width: 100%;
        text-align: center;
        font-size: 0.9rem;
        color: #8892b0;
    }

    /* --- BATCH PREDICTION STYLING --- */
    .stats-card {
        background: rgba(20, 39, 68, 0.5);
        border: 1px solid #303c55;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .stats-card:hover {
        transform: translateY(-3px);
        border-color: #64ffda;
    }
    .stats-card h4 {
        color: #8892b0;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.75rem;
    }
    .stats-card .number {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ccd6f6;
    }
    .stFileUploader > div {
        border: 2px dashed #303c55;
        background-color: rgba(10, 25, 47, 0.5);
        border-radius: 10px;
    }
    .stDownloadButton > button {
        background: linear-gradient(45deg, #64ffda, #00c7a5);
        color: #0a192f !important;
        font-weight: 600 !important;
    }
    .stExpander {
        border: 1px solid #303c55 !important;
        border-radius: 10px !important;
        background: transparent !important;
    }
    .stExpander summary {
        background-color: rgba(20, 39, 68, 0.5) !important;
        color: #64ffda !important;
        font-size: 1.1rem !important;
    }
    .info-box {
        background-color: rgba(100, 255, 218, 0.1);
        border: 1px solid #64ffda;
        border-left: 5px solid #64ffda;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================================
# CONFIGURATION
# ============================================================================
HF_REPO_ID = "IamPradeep/Employee-Churn-Predictor"
MODEL_FILENAME = "final_random_forest_model.joblib"

BEST_FEATURES = [
    "satisfaction_level", "time_spend_company", "average_monthly_hours",
    "number_project", "last_evaluation"
]

FEATURE_DESCRIPTIONS = {
    "satisfaction_level": "Employee satisfaction level (0.0 - 1.0)",
    "time_spend_company": "Years at company (integer)",
    "average_monthly_hours": "Average monthly hours worked (integer)",
    "number_project": "Number of projects (integer)",
    "last_evaluation": "Last performance evaluation score (0.0 - 1.0)"
}

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model_from_huggingface():
    with st.spinner("Syncing with the mothership... 🚀"):
        try:
            model_path = hf_hub_download(
                repo_id=HF_REPO_ID, filename=MODEL_FILENAME, repo_type="model"
            )
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.error(f"❌ Critical Error: Could not load model from the Hugging Face Hub. Details: {str(e)}")
            return None

# ============================================================================
# CALLBACKS FOR SYNCING INPUTS
# ============================================================================
def sync_satisfaction_slider(): st.session_state.satisfaction_level = st.session_state.sat_slider
def sync_satisfaction_input(): st.session_state.satisfaction_level = st.session_state.sat_input
def sync_evaluation_slider(): st.session_state.last_evaluation = st.session_state.eval_slider
def sync_evaluation_input(): st.session_state.last_evaluation = st.session_state.eval_input

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
    return output.getvalue()

# ============================================================================
# INDIVIDUAL PREDICTION TAB
# ============================================================================
def render_individual_prediction_tab(model):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3><i class="fas fa-user-edit"></i> Employee Profile</h3>', unsafe_allow_html=True)
    
    if 'satisfaction_level' not in st.session_state: st.session_state.satisfaction_level = 0.5
    if 'last_evaluation' not in st.session_state: st.session_state.last_evaluation = 0.7
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h6><i class='fas fa-smile' style='color:#64ffda;'></i> Satisfaction Level</h6>", unsafe_allow_html=True)
        satisfaction_level = st.slider("Satisfaction Slider", 0.0, 1.0, st.session_state.satisfaction_level, 0.01, label_visibility="collapsed", key="sat_slider", on_change=sync_satisfaction_slider)
        st.number_input("Satisfaction Input", 0.0, 1.0, st.session_state.satisfaction_level, 0.01, "%.2f", label_visibility="collapsed", key="sat_input", on_change=sync_satisfaction_input)
    
    with col2:
        st.markdown("<h6><i class='fas fa-clipboard-check' style='color:#64ffda;'></i> Last Evaluation Score</h6>", unsafe_allow_html=True)
        last_evaluation = st.slider("Evaluation Slider", 0.0, 1.0, st.session_state.last_evaluation, 0.01, label_visibility="collapsed", key="eval_slider", on_change=sync_evaluation_slider)
        st.number_input("Evaluation Input", 0.0, 1.0, st.session_state.last_evaluation, 0.01, "%.2f", label_visibility="collapsed", key="eval_input", on_change=sync_evaluation_input)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col3, col4, col5 = st.columns(3)
    with col3:
        st.markdown("<h6><i class='fas fa-calendar-alt' style='color:#64ffda;'></i> Years at Company</h6>", unsafe_allow_html=True)
        time_spend_company = st.number_input("Years", 1, 40, 3, 1, label_visibility="collapsed", key="individual_years")
    with col4:
        st.markdown("<h6><i class='fas fa-tasks' style='color:#64ffda;'></i> Number of Projects</h6>", unsafe_allow_html=True)
        number_project = st.number_input("Projects", 1, 10, 4, 1, label_visibility="collapsed", key="individual_projects")
    with col5:
        st.markdown("<h6><i class='fas fa-clock' style='color:#64ffda;'></i> Avg. Monthly Hours</h6>", unsafe_allow_html=True)
        average_monthly_hours = st.number_input("Hours", 80, 350, 200, 5, label_visibility="collapsed", key="individual_hours")

    input_data = {
        'satisfaction_level': st.session_state.satisfaction_level,
        'time_spend_company': time_spend_company,
        'average_monthly_hours': average_monthly_hours,
        'number_project': number_project,
        'last_evaluation': st.session_state.last_evaluation
    }
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🔮 Analyze & Predict Turnover", use_container_width=True, key="individual_predict"):
            input_df = pd.DataFrame([input_data])[BEST_FEATURES]
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            prob_stay = prediction_proba[0]
            prob_leave = prediction_proba[1]
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3><i class="fas fa-chart-line"></i> Prediction Analysis</h3>', unsafe_allow_html=True)

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                if prediction == 0:
                    st.markdown("""
                        <div class="prediction-container">
                            <i class="fas fa-user-check fa-4x" style="color:#50fa7b; margin-bottom: 1rem;"></i>
                            <div class="prediction-result prediction-stay">STAY</div>
                            <p class="prediction-desc">This employee shows strong signs of loyalty.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="prediction-container">
                            <i class="fas fa-user-times fa-4x" style="color:#ff5555; margin-bottom: 1rem;"></i>
                            <div class="prediction-result prediction-leave">LEAVE</div>
                            <p class="prediction-desc">High turnover risk detected for this employee.</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            with res_col2:
                st.markdown("<h5 style='text-align:center;'>Turnover Probability</h5>", unsafe_allow_html=True)
                gauge_class = "gauge-fill-leave" if prediction == 1 else "gauge-fill-stay"
                prob_percent = prob_leave if prediction == 1 else prob_stay
                rotation = prob_percent * 180
                color = "#ff5555" if prediction == 1 else "#50fa7b"

                st.markdown(f"""
                <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
                    <div class="gauge-container">
                        <div class="gauge-bg"></div>
                        <div class="gauge-fill {gauge_class}" style="transform: rotate({rotation}deg);"></div>
                        <div class="gauge-cover"></div>
                        <div class="gauge-text" style="color: {color};">{prob_percent:.0%}</div>
                        <div class="gauge-label">Confidence</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# BATCH PREDICTION TAB
# ============================================================================
def render_batch_prediction_tab(model):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3><i class="fas fa-upload"></i> Upload Employee Data</h3>', unsafe_allow_html=True)
    
    with st.expander("Show Required Data Structure"):
        st.markdown('<div class="info-box">Your uploaded file <strong>must</strong> contain these columns:</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(columns=BEST_FEATURES), use_container_width=True)
        st.info("💡 Tip: Extra columns like 'employee_id' are kept in the output.")

    col1, col2 = st.columns([1, 2])
    with col1:
        file_format = st.selectbox("Select file format", ["CSV", "Excel (.xlsx)"])
    with col2:
        uploader_key = "csv_uploader" if file_format == "CSV" else "excel_uploader"
        file_type = ["csv"] if file_format == "CSV" else ["xlsx", "xls"]
        uploaded_file = st.file_uploader("Drop your file here", type=file_type, key=uploader_key, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) if file_format == "CSV" else pd.read_excel(uploaded_file)
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3><i class="fas fa-cogs"></i> Prediction Settings</h3>', unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                include_probabilities = st.checkbox("Include Probability Scores", value=True, help="Adds 'Probability_Stay' and 'Probability_Leave' columns.")
            with c2:
                include_high_risk_download = st.checkbox("Enable High-Risk Filter Download", value=True, help="Provides a separate download for employees with >50% leave probability.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            missing_cols = [col for col in BEST_FEATURES if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Error: Missing required columns: {', '.join(missing_cols)}. Please check the required data structure.")
            else:
                st.success("✅ File format and columns are correct. Ready for analysis.")
                st.dataframe(df.head(), use_container_width=True)
                
                col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                with col_btn2:
                    if st.button("🚀 Launch Batch Prediction", use_container_width=True, key="batch_predict"):
                        with st.spinner("Analyzing thousands of futures... This may take a moment."):
                            input_features = df[BEST_FEATURES].copy()
                            predictions = model.predict(input_features)
                            prediction_probabilities = model.predict_proba(input_features)
                            
                            result_df = df.copy()
                            result_df['Turnover_Prediction'] = np.where(predictions == 1, 'Leave', 'Stay')
                            if include_probabilities:
                                result_df['Probability_Stay_%'] = (prediction_probabilities[:, 0] * 100).round(2)
                                result_df['Probability_Leave_%'] = (prediction_probabilities[:, 1] * 100).round(2)
                        
                        st.success("✅ Batch prediction complete!")
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.markdown('<h3><i class="fas fa-chart-pie"></i> Results Summary</h3>', unsafe_allow_html=True)

                        leaving_count = sum(predictions == 1)
                        staying_count = sum(predictions == 0)
                        
                        s_col1, s_col2, s_col3 = st.columns(3)
                        s_col1.metric("Total Employees Analyzed", f"{len(df):,}")
                        s_col2.metric("Predicted to Leave ⚠️", f"{leaving_count:,}")
                        s_col3.metric("Predicted to Stay ✅", f"{staying_count:,}")

                        st.markdown("---")
                        st.dataframe(result_df.head(20), use_container_width=True)

                        # Download section
                        st.markdown("---")
                        st.markdown("<h4 style='text-align:center'>Download Your Results</h4>", unsafe_allow_html=True)
                        dl_cols = st.columns(3 if include_high_risk_download else 2)
                        with dl_cols[0]:
                            st.download_button("📥 Download as CSV", convert_df_to_csv(result_df), "predictions.csv", "text/csv", use_container_width=True)
                        with dl_cols[1]:
                            st.download_button("📥 Download as Excel", convert_df_to_excel(result_df), "predictions.xlsx", use_container_width=True)
                        
                        if include_high_risk_download:
                            high_risk_df = result_df[result_df['Probability_Leave_%'] > 50]
                            if not high_risk_df.empty:
                                with dl_cols[2]:
                                    st.download_button(f"⚠️ High-Risk Only ({len(high_risk_df)})", convert_df_to_csv(high_risk_df), "high_risk_employees.csv", use_container_width=True)

                        st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ An error occurred while processing the file: {str(e)}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.markdown('<h1 class="main-header">Employee Turnover Oracle</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Leveraging AI to Forecast Workforce Dynamics & Identify At-Risk Talent</p>', unsafe_allow_html=True)
    
    model = load_model_from_huggingface()
    if model is None:
        st.warning("Model is offline. The application is running in a limited state.")
        return
    
    tab1, tab2 = st.tabs(["Individual Prediction", "Batch Prediction"])
    
    with tab1:
        render_individual_prediction_tab(model)
    with tab2:
        render_batch_prediction_tab(model)

if __name__ == "__main__":
    main()
