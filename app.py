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
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* Center the main content with max-width */
    .block-container {
        max-width: 1200px !important;
        padding-left: 5rem !important;
        padding-right: 5rem !important;
        margin: 0 auto !important;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .stay-prediction {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .leave-prediction {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .feature-box {
        background-color: #e7f3ff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #b8daff;
    }
    
    /* INPUT CARDS - Light Cream Background */
    .feature-card {
        background-color: #FFE8C2;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 5px solid #1E3A5F;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1rem;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }
    
    /* ===== ENHANCED TAB STYLING - MUCH BIGGER TABS WITH BIGGER BOLD TEXT ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        justify-content: center;
        background-color: transparent;
        padding: 10px 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 90px !important;
        min-width: 320px !important;
        padding: 0 50px !important;
        background-color: #f0f2f6;
        border-radius: 15px 15px 0 0 !important;
        font-weight: 900 !important;
        font-size: 1.4rem !important;
        color: #1E3A5F !important;
        border: 2px solid #ddd !important;
        border-bottom: none !important;
        transition: all 0.3s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-transform: none !important;
        letter-spacing: 0.5px !important;
    }
    
    .stTabs [data-baseweb="tab"] p {
        font-weight: 900 !important;
        font-size: 1.4rem !important;
        color: inherit !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e5ec !important;
        transform: translateY(-3px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%) !important;
        color: white !important;
        border: 2px solid #1E3A5F !important;
        border-bottom: none !important;
        box-shadow: 0 4px 20px rgba(30, 58, 95, 0.4) !important;
    }
    
    .stTabs [aria-selected="true"] p {
        color: white !important;
        font-weight: 900 !important;
        font-size: 1.4rem !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 20px;
    }
    
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    
    /* Upload section styling */
    .upload-section {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 2rem;
        border: 2px dashed #1E3A5F;
        margin: 1rem 0;
    }
    
    /* Settings card */
    .settings-card {
        background-color: #e8f4f8;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
    
    /* Column mapping card */
    .mapping-card {
        background-color: #f0e6ff;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #6f42c1;
        margin: 1rem 0;
    }
    
    /* Required columns info box */
    .required-cols-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Success box */
    .success-box {
        background-color: #d4edda;
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Error box */
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #dc3545;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Warning box */
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Stats cards */
    .stats-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #1E3A5F;
    }
    .stats-card h3 {
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .stats-card .number {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A5F;
    }
    
    /* ===== VIBRANT RAINBOW GRADIENT PREDICT BUTTON ===== */
    .stButton>button {
        width: 100%;
        background: linear-gradient(
            45deg, 
            #ff0080, #ff8c00, #40e0d0, #ff0080, #ff8c00
        );
        background-size: 400% 400%;
        color: white !important;
        font-size: 1.4rem;
        font-weight: 900 !important;
        padding: 1.2rem 2.5rem;
        border-radius: 50px;
        border: none !important;
        outline: none !important;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 4px 15px rgba(255, 0, 128, 0.4),
            0 8px 30px rgba(255, 140, 0, 0.3),
            0 0 40px rgba(64, 224, 208, 0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: gradientShift 3s ease infinite, pulse 2s ease-in-out infinite;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Gradient animation - shifting colors */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Pulse glow animation */
    @keyframes pulse {
        0% { 
            box-shadow: 
                0 4px 15px rgba(255, 0, 128, 0.4),
                0 8px 30px rgba(255, 140, 0, 0.3),
                0 0 40px rgba(64, 224, 208, 0.2);
            transform: scale(1);
        }
        50% { 
            box-shadow: 
                0 6px 25px rgba(255, 0, 128, 0.6),
                0 12px 40px rgba(255, 140, 0, 0.5),
                0 0 60px rgba(64, 224, 208, 0.4),
                0 0 80px rgba(255, 0, 128, 0.2);
            transform: scale(1.02);
        }
        100% { 
            box-shadow: 
                0 4px 15px rgba(255, 0, 128, 0.4),
                0 8px 30px rgba(255, 140, 0, 0.3),
                0 0 40px rgba(64, 224, 208, 0.2);
            transform: scale(1);
        }
    }
    
    /* Hover - Electric effect with different gradient */
    .stButton>button:hover {
        background: linear-gradient(
            45deg, 
            #00f5ff, #ff00ff, #ffff00, #00f5ff, #ff00ff
        );
        background-size: 400% 400%;
        transform: translateY(-5px) scale(1.01);
        box-shadow: 
            0 10px 30px rgba(0, 245, 255, 0.5),
            0 15px 50px rgba(255, 0, 255, 0.4),
            0 0 100px rgba(255, 255, 0, 0.3),
            inset 0 0 20px rgba(255, 255, 255, 0.1);
        animation: gradientShift 1.5s ease infinite;
        color: white !important;
        border: none !important;
        outline: none !important;
        font-weight: 900 !important;
    }
    
    /* Active/Click - Neon burst effect */
    .stButton>button:active {
        background: linear-gradient(
            45deg, 
            #ff3366, #ff6b35, #f7931e, #ffd700, #ff3366
        );
        background-size: 400% 400%;
        transform: translateY(2px) scale(0.98);
        box-shadow: 
            0 2px 10px rgba(255, 51, 102, 0.6),
            0 4px 20px rgba(255, 107, 53, 0.4),
            inset 0 0 30px rgba(255, 255, 255, 0.2);
        color: white !important;
        border: none !important;
        outline: none !important;
        font-weight: 900 !important;
    }
    
    /* Shimmer/shine effect overlay */
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            120deg,
            transparent,
            rgba(255, 255, 255, 0.4),
            transparent
        );
        transition: left 0.7s ease;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    /* Sparkle particles effect */
    .stButton>button::after {
        content: '✨';
        position: absolute;
        font-size: 1.2rem;
        right: 20px;
        animation: sparkle 1.5s ease-in-out infinite;
    }
    
    @keyframes sparkle {
        0%, 100% { opacity: 1; transform: scale(1) rotate(0deg); }
        50% { opacity: 0.5; transform: scale(1.3) rotate(180deg); }
    }
    
    /* Focus effect - Remove blue border completely */
    .stButton>button:focus {
        outline: none !important;
        border: none !important;
        box-shadow: 
            0 4px 15px rgba(255, 0, 128, 0.4),
            0 8px 30px rgba(255, 140, 0, 0.3),
            0 0 40px rgba(64, 224, 208, 0.2);
        color: white !important;
        font-weight: 900 !important;
    }
    
    /* Focus-visible - Remove blue border completely */
    .stButton>button:focus-visible {
        outline: none !important;
        border: none !important;
        box-shadow: 
            0 4px 15px rgba(255, 0, 128, 0.4),
            0 8px 30px rgba(255, 140, 0, 0.3),
            0 0 40px rgba(64, 224, 208, 0.2);
        font-weight: 900 !important;
    }
    
    /* Remove focus ring from button container as well */
    .stButton>button:focus:not(:focus-visible) {
        outline: none !important;
        border: none !important;
    }
    
    /* Ensure text stays white in ALL states */
    .stButton>button,
    .stButton>button:hover,
    .stButton>button:active,
    .stButton>button:focus,
    .stButton>button:focus-visible,
    .stButton>button:visited,
    .stButton>button span,
    .stButton>button:hover span,
    .stButton>button:active span,
    .stButton>button:focus span,
    .stButton>button p,
    .stButton>button:hover p,
    .stButton>button:active p,
    .stButton>button:focus p,
    .stButton>button div,
    .stButton>button:hover div,
    .stButton>button:active div,
    .stButton>button:focus div {
        color: white !important;
        outline: none !important;
        border: none !important;
        font-weight: 900 !important;
    }
    
    /* Download button styling */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #28a745, #20c997) !important;
        animation: none !important;
    }
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #20c997, #28a745) !important;
    }
    
    .info-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .progress-bar-container {
        width: 100%;
        background-color: #e9ecef;
        border-radius: 10px;
        margin: 5px 0 15px 0;
        height: 25px;
        overflow: hidden;
    }
    .progress-bar-green {
        height: 100%;
        background-color: #28a745;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }
    .progress-bar-red {
        height: 100%;
        background-color: #dc3545;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }
    
    /* Blue styled expander */
    div[data-testid="stExpander"] {
        border: none !important;
        border-radius: 8px !important;
    }
    div[data-testid="stExpander"] details {
        border: none !important;
    }
    div[data-testid="stExpander"] details summary {
        background-color: #1E3A5F !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        font-size: 1.2rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stExpander"] details summary:hover {
        background-color: #2E5A8F !important;
        color: white !important;
    }
    div[data-testid="stExpander"] details summary svg {
        color: white !important;
        fill: white !important;
    }
    div[data-testid="stExpander"] details[open] summary {
        border-radius: 8px 8px 0 0 !important;
    }
    div[data-testid="stExpander"] details > div {
        border: 1px solid #1E3A5F !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }
    
    /* Checkbox styling */
    .stCheckbox {
        padding: 0.5rem 0;
    }
    .stCheckbox label {
        font-size: 1rem;
        font-weight: 500;
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

# Feature descriptions for user reference
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
# CALLBACK FUNCTIONS FOR SYNCING SLIDERS AND NUMBER INPUTS
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

def get_default_index(feature, column_list):
    """Find the best matching column index for a feature"""
    if feature in column_list:
        return column_list.index(feature)
    for i, col in enumerate(column_list):
        if feature.lower() in col.lower() or col.lower() in feature.lower():
            return i
    return 0

# ============================================================================
# INDIVIDUAL PREDICTION TAB
# ============================================================================
def render_individual_prediction_tab(model):
    """Render the individual prediction tab content"""
    
    st.markdown("---")
    st.markdown('<h2 class="section-header">📝 Enter Employee Information</h2>', unsafe_allow_html=True)
    
    if 'satisfaction_level' not in st.session_state:
        st.session_state.satisfaction_level = 0.5
    if 'last_evaluation' not in st.session_state:
        st.session_state.last_evaluation = 0.7
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <span style="font-size: 1.2rem;">😊 <strong>Satisfaction Level</strong></span>
            </div>
            """, unsafe_allow_html=True)
            
            sat_col1, sat_col2 = st.columns([3, 1])
            with sat_col1:
                st.slider(
                    "Satisfaction Slider",
                    min_value=0.0, max_value=1.0,
                    value=st.session_state.satisfaction_level,
                    step=0.01, label_visibility="collapsed",
                    key="sat_slider", on_change=sync_satisfaction_slider
                )
            with sat_col2:
                st.number_input(
                    "Satisfaction Input",
                    min_value=0.0, max_value=1.0,
                    value=st.session_state.satisfaction_level,
                    step=0.01, format="%.2f", label_visibility="collapsed",
                    key="sat_input", on_change=sync_satisfaction_input
                )
            satisfaction_level = st.session_state.satisfaction_level
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <span style="font-size: 1.2rem;">📊 <strong>Last Evaluation</strong></span>
            </div>
            """, unsafe_allow_html=True)
            
            eval_col1, eval_col2 = st.columns([3, 1])
            with eval_col1:
                st.slider(
                    "Evaluation Slider",
                    min_value=0.0, max_value=1.0,
                    value=st.session_state.last_evaluation,
                    step=0.01, label_visibility="collapsed",
                    key="eval_slider", on_change=sync_evaluation_slider
                )
            with eval_col2:
                st.number_input(
                    "Evaluation Input",
                    min_value=0.0, max_value=1.0,
                    value=st.session_state.last_evaluation,
                    step=0.01, format="%.2f", label_visibility="collapsed",
                    key="eval_input", on_change=sync_evaluation_input
                )
            last_evaluation = st.session_state.last_evaluation
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <span style="font-size: 1.2rem;">📅 <strong>Years at Company</strong></span>
            </div>
            """, unsafe_allow_html=True)
            time_spend_company = st.number_input(
                "Years", min_value=1, max_value=40, value=3, step=1,
                label_visibility="collapsed", key="individual_years"
            )
        
        with col4:
            st.markdown("""
            <div class="feature-card">
                <span style="font-size: 1.2rem;">📁 <strong>Number of Projects</strong></span>
            </div>
            """, unsafe_allow_html=True)
            number_project = st.number_input(
                "Projects", min_value=1, max_value=10, value=4, step=1,
                label_visibility="collapsed", key="individual_projects"
            )
        
        with col5:
            st.markdown("""
            <div class="feature-card">
                <span style="font-size: 1.2rem;">⏰ <strong>Avg. Monthly Hours</strong></span>
            </div>
            """, unsafe_allow_html=True)
            average_monthly_hours = st.number_input(
                "Hours", min_value=80, max_value=350, value=200, step=5,
                label_visibility="collapsed", key="individual_hours"
            )
    
    input_data = {
        'satisfaction_level': satisfaction_level,
        'time_spend_company': time_spend_company,
        'average_monthly_hours': average_monthly_hours,
        'number_project': number_project,
        'last_evaluation': last_evaluation
    }
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Employee Turnover", use_container_width=True, key="individual_predict")
    
    if predict_button:
        input_df = pd.DataFrame([input_data])[BEST_FEATURES]
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        prob_stay = prediction_proba[0] * 100
        prob_leave = prediction_proba[1] * 100
        
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0:
                st.markdown("""
                <div class="prediction-box stay-prediction">
                    <h1>✅ STAY</h1>
                    <p style="font-size: 1.3rem; margin-top: 1rem;">
                        Employee is likely to <strong>STAY</strong> with the company
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-box leave-prediction">
                    <h1>⚠️ LEAVE</h1>
                    <p style="font-size: 1.3rem; margin-top: 1rem;">
                        Employee is likely to <strong>LEAVE</strong> the company
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Prediction Probabilities")
            st.write(f"**Probability of Staying:** {prob_stay:.1f}%")
            st.markdown(f"""
            <div class="progress-bar-container">
                <div class="progress-bar-green" style="width: {prob_stay}%;"></div>
            </div>
            """, unsafe_allow_html=True)
            st.write(f"**Probability of Leaving:** {prob_leave:.1f}%")
            st.markdown(f"""
            <div class="progress-bar-container">
                <div class="progress-bar-red" style="width: {prob_leave}%;"></div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# BATCH PREDICTION TAB
# ============================================================================
def render_batch_prediction_tab(model):
    """Render the batch prediction tab content"""
    
    st.markdown("---")
    st.markdown('<h2 class="section-header">📊 Batch Employee Prediction</h2>', unsafe_allow_html=True)
    
    with st.expander("📋 Required Columns in Your File (Click to Expand)"):
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <p>Your uploaded file <strong>must contain</strong> these columns (or you can map your columns using Column Mapping):</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            for feature in BEST_FEATURES[:3]:
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 0.8rem; border-radius: 6px; margin: 0.5rem 0; border-left: 4px solid #1E3A5F;">
                    <strong>{feature}</strong><br/>
                    <small style="color: #666;">{FEATURE_DESCRIPTIONS[feature]}</small>
                </div>
                """, unsafe_allow_html=True)
        with col2:
            for feature in BEST_FEATURES[3:]:
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 0.8rem; border-radius: 6px; margin: 0.5rem 0; border-left: 4px solid #1E3A5F;">
                    <strong>{feature}</strong><br/>
                    <small style="color: #666;">{FEATURE_DESCRIPTIONS[feature]}</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.info("💡 **Tip:** Additional columns will be preserved in the output but won't be used for prediction.")
    
    st.markdown("---")
    
    # File Upload Section
    st.markdown("### 📁 Upload Your Data")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""<div class="settings-card"><h4>⚙️ File Settings</h4></div>""", unsafe_allow_html=True)
        file_format = st.selectbox("Select file format", options=["CSV", "Excel (.xlsx)"], index=0, key="file_format")
    
    with col2:
        if file_format == "CSV":
            uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="csv_uploader")
        else:
            uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"], key="excel_uploader")
    
    st.markdown("---")
    
    # Prediction Output Settings
    st.markdown("### ⚙️ Prediction Output Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""<div class="settings-card"><h4>📝 Column Name for Predictions</h4></div>""", unsafe_allow_html=True)
        column_name_option = st.selectbox(
            "Select prediction column name",
            options=["Prediction", "Churn", "Will_Leave", "Turnover", "Custom"],
            index=0, key="column_name_option"
        )
        if column_name_option == "Custom":
            custom_column_name = st.text_input("Enter custom column name", value="My_Prediction", key="custom_column_name")
            prediction_column_name = custom_column_name if custom_column_name.strip() else "Prediction"
        else:
            prediction_column_name = column_name_option
    
    with col2:
        st.markdown("""<div class="settings-card"><h4>🏷️ Prediction Labels</h4></div>""", unsafe_allow_html=True)
        label_option = st.selectbox(
            "Select prediction labels",
            options=["Leave / Stay", "Yes / No", "Churn / Not Churn", "1 / 0", "True / False", "Custom"],
            index=0, key="label_option"
        )
        
        label_mappings = {
            "Leave / Stay": {1: "Leave", 0: "Stay"},
            "Yes / No": {1: "Yes", 0: "No"},
            "Churn / Not Churn": {1: "Churn", 0: "Not Churn"},
            "1 / 0": {1: "1", 0: "0"},
            "True / False": {1: "True", 0: "False"}
        }
        
        if label_option == "Custom":
            custom_col1, custom_col2 = st.columns(2)
            with custom_col1:
                custom_leave_label = st.text_input("Label for LEAVING", value="Leaving", key="custom_leave_label")
            with custom_col2:
                custom_stay_label = st.text_input("Label for STAYING", value="Staying", key="custom_stay_label")
            prediction_labels = {
                1: custom_leave_label if custom_leave_label.strip() else "Leaving",
                0: custom_stay_label if custom_stay_label.strip() else "Staying"
            }
        else:
            prediction_labels = label_mappings[label_option]
        
        st.info(f"📌 **Label Preview:** Leaving → '{prediction_labels[1]}' | Staying → '{prediction_labels[0]}'")
    
    st.markdown("---")
    st.markdown("### 📊 Additional Output Options")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""<div class="settings-card"><h4>🎯 Probability Columns</h4></div>""", unsafe_allow_html=True)
        include_probabilities = st.checkbox("Include prediction probabilities in output", value=True, key="include_probabilities")
        if include_probabilities:
            st.success("✅ Two additional columns will be added: `Probability_Stay` and `Probability_Leave`")
    
    with col2:
        st.markdown("""<div class="settings-card"><h4>⚠️ High Risk Filter</h4></div>""", unsafe_allow_html=True)
        include_high_risk_download = st.checkbox("Enable high-risk employees download", value=True, key="include_high_risk")
        if include_high_risk_download:
            st.success("✅ A separate download button for high-risk employees will be available")
    
    # Process Uploaded File
    if uploaded_file is not None:
        st.markdown("---")
        st.markdown("### 📄 Uploaded Data Preview")
        
        try:
            if file_format == "CSV":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # File info stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""<div class="stats-card"><h4>Total Rows</h4><div class="number">{len(df):,}</div></div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class="stats-card"><h4>Total Columns</h4><div class="number">{len(df.columns)}</div></div>""", unsafe_allow_html=True)
            with col3:
                available_features = [col for col in BEST_FEATURES if col in df.columns]
                st.markdown(f"""<div class="stats-card"><h4>Required Cols Found</h4><div class="number">{len(available_features)}/{len(BEST_FEATURES)}</div></div>""", unsafe_allow_html=True)
            
            st.dataframe(df.head(10), use_container_width=True)
            
            missing_columns = [col for col in BEST_FEATURES if col not in df.columns]
            
            # ========================================================================
            # SIMPLIFIED COLUMN MAPPING SECTION
            # ========================================================================
            st.markdown("---")
            
            enable_column_mapping = st.checkbox(
                "🔄 **Enable Column 
