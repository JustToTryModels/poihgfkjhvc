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
    
    /* ===== ENHANCED TAB STYLING ===== */
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
    
    /* Mapping card */
    .mapping-card {
        background-color: #fff8e1;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #ffc107;
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
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes pulse {
        0% { 
            box-shadow: 0 4px 15px rgba(255, 0, 128, 0.4), 0 8px 30px rgba(255, 140, 0, 0.3), 0 0 40px rgba(64, 224, 208, 0.2);
            transform: scale(1);
        }
        50% { 
            box-shadow: 0 6px 25px rgba(255, 0, 128, 0.6), 0 12px 40px rgba(255, 140, 0, 0.5), 0 0 60px rgba(64, 224, 208, 0.4), 0 0 80px rgba(255, 0, 128, 0.2);
            transform: scale(1.02);
        }
        100% { 
            box-shadow: 0 4px 15px rgba(255, 0, 128, 0.4), 0 8px 30px rgba(255, 140, 0, 0.3), 0 0 40px rgba(64, 224, 208, 0.2);
            transform: scale(1);
        }
    }
    
    /* Hover/Active states for button */
    .stButton>button:hover {
        background: linear-gradient(45deg, #00f5ff, #ff00ff, #ffff00, #00f5ff, #ff00ff);
        background-size: 400% 400%;
        transform: translateY(-5px) scale(1.01);
        animation: gradientShift 1.5s ease infinite;
        color: white !important;
    }
    
    .stButton>button:active {
        transform: translateY(2px) scale(0.98);
        color: white !important;
    }
    
    .stButton>button:focus, .stButton>button:focus-visible {
        outline: none !important;
        border: none !important;
        color: white !important;
    }
    
    /* Download button styling */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #28a745, #20c997) !important;
        animation: none !important;
    }
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #20c997, #28a745) !important;
    }
    
    /* Progress bars */
    .progress-bar-container {
        width: 100%;
        background-color: #e9ecef;
        border-radius: 10px;
        margin: 5px 0 15px 0;
        height: 25px;
        overflow: hidden;
    }
    .progress-bar-green { height: 100%; background-color: #28a745; transition: width 0.5s ease-in-out; }
    .progress-bar-red { height: 100%; background-color: #dc3545; transition: width 0.5s ease-in-out; }
    
    /* Expander styling */
    div[data-testid="stExpander"] { border: none !important; border-radius: 8px !important; }
    div[data-testid="stExpander"] details summary {
        background-color: #1E3A5F !important; color: white !important; border-radius: 8px !important; padding: 0.75rem 1rem !important;
    }
    div[data-testid="stExpander"] details > div {
        border: 1px solid #1E3A5F !important; border-top: none !important; border-radius: 0 0 8px 8px !important;
    }
    
    /* Success/Error boxes */
    .success-box { background-color: #d4edda; border: 1px solid #28a745; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
    .error-box { background-color: #f8d7da; border: 1px solid #dc3545; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
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
    """Render the individual prediction tab content"""
    st.markdown("---")
    st.markdown('<h2 class="section-header">📝 Enter Employee Information</h2>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'satisfaction_level' not in st.session_state: st.session_state.satisfaction_level = 0.5
    if 'last_evaluation' not in st.session_state: st.session_state.last_evaluation = 0.7
    
    # Define sync functions
    def sync_satisfaction_slider(): st.session_state.satisfaction_level = st.session_state.sat_slider
    def sync_satisfaction_input(): st.session_state.satisfaction_level = st.session_state.sat_input
    def sync_evaluation_slider(): st.session_state.last_evaluation = st.session_state.eval_slider
    def sync_evaluation_input(): st.session_state.last_evaluation = st.session_state.eval_input
    
    # Input Forms
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="feature-card"><span style="font-size: 1.2rem;">😊 <strong>Satisfaction Level</strong></span></div>', unsafe_allow_html=True)
            sat_col1, sat_col2 = st.columns([3, 1])
            with sat_col1: st.slider("Sat Slider", 0.0, 1.0, st.session_state.satisfaction_level, 0.01, label_visibility="collapsed", key="sat_slider", on_change=sync_satisfaction_slider)
            with sat_col2: st.number_input("Sat Input", 0.0, 1.0, st.session_state.satisfaction_level, 0.01, label_visibility="collapsed", key="sat_input", on_change=sync_satisfaction_input)
            
        with col2:
            st.markdown('<div class="feature-card"><span style="font-size: 1.2rem;">📊 <strong>Last Evaluation</strong></span></div>', unsafe_allow_html=True)
            eval_col1, eval_col2 = st.columns([3, 1])
            with eval_col1: st.slider("Eval Slider", 0.0, 1.0, st.session_state.last_evaluation, 0.01, label_visibility="collapsed", key="eval_slider", on_change=sync_evaluation_slider)
            with eval_col2: st.number_input("Eval Input", 0.0, 1.0, st.session_state.last_evaluation, 0.01, label_visibility="collapsed", key="eval_input", on_change=sync_evaluation_input)
        
        col3, col4, col5 = st.columns(3)
        with col3:
            st.markdown('<div class="feature-card"><span style="font-size: 1.2rem;">📅 <strong>Years at Company</strong></span></div>', unsafe_allow_html=True)
            time_spend_company = st.number_input("Years", 1, 40, 3, 1, label_visibility="collapsed", key="individual_years")
        with col4:
            st.markdown('<div class="feature-card"><span style="font-size: 1.2rem;">📁 <strong>Number of Projects</strong></span></div>', unsafe_allow_html=True)
            number_project = st.number_input("Projects", 1, 10, 4, 1, label_visibility="collapsed", key="individual_projects")
        with col5:
            st.markdown('<div class="feature-card"><span style="font-size: 1.2rem;">⏰ <strong>Avg. Monthly Hours</strong></span></div>', unsafe_allow_html=True)
            average_monthly_hours = st.number_input("Hours", 80, 350, 200, 5, label_visibility="collapsed", key="individual_hours")

    st.markdown("---")
    _, col2, _ = st.columns([1, 2, 1])
    with col2: predict_button = st.button("🔮 Predict Employee Turnover", use_container_width=True, key="individual_predict")
    
    if predict_button:
        input_data = pd.DataFrame([{
            'satisfaction_level': st.session_state.satisfaction_level,
            'time_spend_company': time_spend_company,
            'average_monthly_hours': average_monthly_hours,
            'number_project': number_project,
            'last_evaluation': st.session_state.last_evaluation
        }])[BEST_FEATURES]
        
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 0:
                st.markdown('<div class="prediction-box stay-prediction"><h1>✅ STAY</h1><p style="font-size: 1.3rem;">Likely to <strong>STAY</strong></p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-box leave-prediction"><h1>⚠️ LEAVE</h1><p style="font-size: 1.3rem;">Likely to <strong>LEAVE</strong></p></div>', unsafe_allow_html=True)
        with col2:
            st.write(f"**Probability of Staying:** {prediction_proba[0]*100:.1f}%")
            st.markdown(f'<div class="progress-bar-container"><div class="progress-bar-green" style="width: {prediction_proba[0]*100}%;"></div></div>', unsafe_allow_html=True)
            st.write(f"**Probability of Leaving:** {prediction_proba[1]*100:.1f}%")
            st.markdown(f'<div class="progress-bar-container"><div class="progress-bar-red" style="width: {prediction_proba[1]*100}%;"></div></div>', unsafe_allow_html=True)

# ============================================================================
# BATCH PREDICTION TAB
# ============================================================================
def render_batch_prediction_tab(model):
    """Render the batch prediction tab content"""
    
    st.markdown("---")
    st.markdown('<h2 class="section-header">📊 Batch Employee Prediction</h2>', unsafe_allow_html=True)
    
    with st.expander("📋 Required Columns (Click to Expand)"):
        st.info("Your file should contain these data points. If your column names are different, you can map them after uploading.")
        cols = st.columns(5)
        for i, feature in enumerate(BEST_FEATURES):
            with cols[i]:
                st.markdown(f"**{feature}**")
                st.caption(FEATURE_DESCRIPTIONS[feature])
    
    st.markdown("---")
    
    # 1. FILE UPLOAD
    st.markdown("### 📁 Upload Your Data")
    col1, col2 = st.columns([1, 2])
    with col1:
        file_format = st.selectbox("Select file format", ["CSV", "Excel (.xlsx)"], key="file_format")
    with col2:
        if file_format == "CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader")
        else:
            uploaded_file = st.file_uploader("Upload Excel", type=["xlsx", "xls"], key="excel_uploader")
    
    # 2. OUTPUT SETTINGS
    st.markdown("---")
    st.markdown("### ⚙️ Output Settings")
    col1, col2 = st.columns(2)
    with col1:
        col_name_opt = st.selectbox("Result Column Name", ["Prediction", "Churn", "Will_Leave", "Custom"], key="col_opt")
        prediction_column_name = st.text_input("Custom Name", "My_Prediction") if col_name_opt == "Custom" else col_name_opt
    with col2:
        include_probs = st.checkbox("Include Probabilities", value=True, key="inc_prob")
    
    # 3. PROCESS FILE
    if uploaded_file is not None:
        st.markdown("---")
        st.markdown("### 📄 Data Preview & Mapping")
        
        try:
            if file_format == "CSV":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Show initial stats
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Rows", len(df))
            c2.metric("Total Columns", len(df.columns))
            
            st.dataframe(df.head(5), use_container_width=True)
            
            # ------------------------------------------------------------------
            # COLUMN MAPPING SECTION (New Feature)
            # ------------------------------------------------------------------
            
            # Check for missing columns
            missing_columns = [col for col in BEST_FEATURES if col not in df.columns]
            
            # Flag to track if mapping is active
            mapping_active = False
            column_mapping = {} # Store user choices: {model_feature_name: user_column_name}
            
            # If columns are missing, force open mapping. Else allow user to open it.
            if missing_columns:
                st.markdown(f"""
                <div class="error-box">
                    <h4>⚠️ Column Mapping Required</h4>
                    <p>Some required columns were not found automatically. Please match your columns to the required features below.</p>
                </div>
                """, unsafe_allow_html=True)
                mapping_expander_expanded = True
            else:
                mapping_expander_expanded = False
                
            # Option to enable mapping even if columns match (in case names are same but mean different things)
            use_custom_mapping = st.checkbox("🔄 My columns have different names (Enable Column Rename/Mapping)", value=bool(missing_columns))
            
            if use_custom_mapping or missing_columns:
                mapping_active = True
                with st.expander("🛠️ Map Your Columns (Temporary Rename for Prediction)", expanded=True):
                    st.markdown("""
                    <div class="mapping-card">
                        <p>Select which column from your uploaded file corresponds to the model's required feature. 
                        <strong>This will not change your actual data file.</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    map_cols = st.columns(3)
                    
                    # Create dropdowns for each required feature
                    for i, feature in enumerate(BEST_FEATURES):
                        with map_cols[i % 3]:
                            # Try to auto-select if name matches
                            try:
                                default_idx = list(df.columns).index(feature)
                            except ValueError:
                                default_idx = 0
                                
                            selected_col = st.selectbox(
                                f"Map to: **{feature}**",
                                options=df.columns,
                                index=default_idx,
                                help=f"Select the column in your file that represents '{feature}'",
                                key=f"map_{feature}"
                            )
                            column_mapping[feature] = selected_col
            else:
                # If mapping not active, assume exact match
                for feature in BEST_FEATURES:
                    column_mapping[feature] = feature

            # ------------------------------------------------------------------
            # VALIDATION & PREDICTION
            # ------------------------------------------------------------------
            
            # Validate that we have all features mapped or present
            ready_to_predict = True
            if not mapping_active and missing_columns:
                ready_to_predict = False # Should be handled by logic above, but safe check
            
            if ready_to_predict:
                st.markdown("---")
                _, btn_col, _ = st.columns([1, 2, 1])
                with btn_col:
                    batch_predict_button = st.button("🔮 Generate Batch Predictions", use_container_width=True)
                
                if batch_predict_button:
                    with st.spinner("🔄 Processing predictions with temporary column mapping..."):
                        
                        # 1. Prepare temporary dataframe for prediction
                        # Create a new empty dataframe
                        prediction_input_df = pd.DataFrame()
                        
                        try:
                            # Fill it using the mapping
                            for model_feature, user_column in column_mapping.items():
                                prediction_input_df[model_feature] = df[user_column]
                            
                            # Ensure correct order of columns for the model
                            prediction_input_df = prediction_input_df[BEST_FEATURES]
                            
                            # 2. Make Predictions
                            predictions = model.predict(prediction_input_df)
                            probs = model.predict_proba(prediction_input_df)
                            
                            # 3. Create Result DataFrame (Copy ORIGINAL df to preserve user columns)
                            result_df = df.copy()
                            
                            # Add results
                            labels = {0: "Stay", 1: "Leave"}
                            result_df[prediction_column_name] = [labels[p] for p in predictions]
                            
                            if include_probs:
                                result_df[f"{prediction_column_name}_Prob_Stay"] = (probs[:, 0] * 100).round(2)
                                result_df[f"{prediction_column_name}_Prob_Leave"] = (probs[:, 1] * 100).round(2)
                            
                            # Success Message
                            st.success("✅ Predictions generated successfully!")
                            
                            # 4. Show Results & Download
                            st.markdown("### 📊 Results Summary")
                            
                            # Calculate stats
                            n_leave = sum(predictions == 1)
                            pct_leave = (n_leave / len(predictions)) * 100
                            
                            kpi1, kpi2, kpi3 = st.columns(3)
                            kpi1.metric("Total Employees", len(predictions))
                            kpi2.metric("Predicted to Leave", f"{n_leave} ({pct_leave:.1f}%)")
                            kpi3.metric("Predicted to Stay", f"{len(predictions)-n_leave} ({100-pct_leave:.1f}%)")
                            
                            st.dataframe(result_df.head(10), use_container_width=True)
                            
                            d_col1, d_col2 = st.columns(2)
                            with d_col1:
                                st.download_button(
                                    "📥 Download CSV", 
                                    data=convert_df_to_csv(result_df), 
                                    file_name="predictions.csv", 
                                    mime="text/csv", 
                                    use_container_width=True
                                )
                            with d_col2:
                                st.download_button(
                                    "📥 Download Excel", 
                                    data=convert_df_to_excel(result_df), 
                                    file_name="predictions.xlsx", 
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                                    use_container_width=True
                                )
                                
                        except Exception as e:
                            st.error(f"❌ Error during prediction: {str(e)}")
                            st.warning("Please check if the selected columns contain numeric data compatible with the model.")

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.markdown('<h1 class="main-header">👥 Employee Turnover Prediction</h1>', unsafe_allow_html=True)
    
    model = load_model_from_huggingface()
    if model is None:
        st.error("Failed to load model from Hugging Face.")
        return
    
    tab1, tab2 = st.tabs(["📝 Individual Prediction", "📊 Batch Prediction"])
    
    with tab1: render_individual_prediction_tab(model)
    with tab2: render_batch_prediction_tab(model)

if __name__ == "__main__":
    main()
