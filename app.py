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
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%) !important;
        color: white !important;
        border: 2px solid #1E3A5F !important;
        border-bottom: none !important;
        box-shadow: 0 4px 20px rgba(30, 58, 95, 0.4) !important;
    }
    
    /* Settings card */
    .settings-card {
        background-color: #e8f4f8;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
    
    /* Mapping box */
    .mapping-box {
        background-color: #f0e6ff;
        border: 1px solid #6f42c1;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Success/Error boxes */
    .success-box {
        background-color: #d4edda;
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #dc3545;
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
    .stats-card .number {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A5F;
    }
    
    /* ===== PREDICT BUTTON ===== */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #ff0080, #ff8c00, #40e0d0, #ff0080);
        background-size: 400% 400%;
        color: white !important;
        font-size: 1.4rem;
        font-weight: 900 !important;
        padding: 1.2rem 2.5rem;
        border-radius: 50px;
        border: none !important;
        animation: gradientShift 3s ease infinite;
        text-transform: uppercase;
        letter-spacing: 3px;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #28a745, #20c997) !important;
        animation: none !important;
    }

    .progress-bar-container {
        width: 100%;
        background-color: #e9ecef;
        border-radius: 10px;
        margin: 5px 0 15px 0;
        height: 25px;
        overflow: hidden;
    }
    .progress-bar-green { height: 100%; background-color: #28a745; transition: width 0.5s; }
    .progress-bar-red { height: 100%; background-color: #dc3545; transition: width 0.5s; }
</style>
""", unsafe_allow_html=True)

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
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model_from_huggingface():
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME, repo_type="model")
        return joblib.load(model_path)
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

def get_compact_column_mapping(df_columns, enable_mapping):
    """
    Renders a compact, editable dataframe for column mapping.
    """
    if not enable_mapping:
        # Default: map features to themselves
        return {f: f for f in BEST_FEATURES}, True

    st.markdown("""
    <div class="mapping-box">
        <h4 style="margin:0; color:#6f42c1;">🔄 Map Data Columns</h4>
        <p style="margin:5px 0 0 0; color:#666; font-size:0.9rem;">
            Select which column in your file matches the required model feature. 
            <strong>Predictions only. Original data remains unchanged.</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Prepare initial data for the editor
    mapping_data = []
    user_cols_list = list(df_columns)
    
    for feature in BEST_FEATURES:
        # Auto-detect best match
        current_match = user_cols_list[0] # Default to first
        
        # Check for exact match
        if feature in user_cols_list:
            current_match = feature
        else:
            # Check for partial match
            for col in user_cols_list:
                if feature.lower() in col.lower() or col.lower() in feature.lower():
                    current_match = col
                    break
        
        mapping_data.append({
            "Required Feature": feature,
            "Your File Column": current_match,
            "Description": FEATURE_DESCRIPTIONS[feature]
        })

    mapping_df = pd.DataFrame(mapping_data)

    # 2. Render the Editable Dataframe
    edited_df = st.data_editor(
        mapping_df,
        column_config={
            "Required Feature": st.column_config.TextColumn(
                "Model Feature", 
                help="The feature name the model expects",
                disabled=True
            ),
            "Your File Column": st.column_config.SelectboxColumn(
                "Map to Your Column",
                help="Select the column from your uploaded file",
                width="medium",
                options=user_cols_list,
                required=True
            ),
            "Description": st.column_config.TextColumn(
                "Description",
                disabled=True,
                width="large"
            )
        },
        hide_index=True,
        use_container_width=True,
        key="mapping_editor"
    )

    # 3. Convert back to dictionary and Validate
    final_mapping = dict(zip(edited_df["Required Feature"], edited_df["Your File Column"]))
    
    # Validation: Check for duplicates
    used_cols = list(final_mapping.values())
    duplicates = [col for col in set(used_cols) if used_cols.count(col) > 1]
    
    if duplicates:
        st.markdown(f"""
        <div class="error-box" style="padding: 0.5rem 1rem;">
            ⚠️ <strong>Error:</strong> You have mapped '<strong>{', '.join(duplicates)}</strong>' to multiple features. Please ensure each feature uses a unique column.
        </div>
        """, unsafe_allow_html=True)
        return final_mapping, False
    
    return final_mapping, True

# ============================================================================
# INDIVIDUAL PREDICTION TAB
# ============================================================================
def render_individual_prediction_tab(model):
    st.markdown("---")
    st.markdown('<h2 class="section-header">📝 Enter Employee Information</h2>', unsafe_allow_html=True)
    
    if 'satisfaction_level' not in st.session_state: st.session_state.satisfaction_level = 0.5
    if 'last_evaluation' not in st.session_state: st.session_state.last_evaluation = 0.7

    # Row 1
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="feature-card"><span>😊 <strong>Satisfaction Level</strong></span></div>', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1])
        c1.slider("Sat Slider", 0.0, 1.0, key="sat_slider", step=0.01, label_visibility="collapsed", on_change=lambda: st.session_state.update(satisfaction_level=st.session_state.sat_slider))
        c2.number_input("Sat Input", 0.0, 1.0, key="sat_input", step=0.01, label_visibility="collapsed", on_change=lambda: st.session_state.update(satisfaction_level=st.session_state.sat_input))
    
    with col2:
        st.markdown('<div class="feature-card"><span>📊 <strong>Last Evaluation</strong></span></div>', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1])
        c1.slider("Eval Slider", 0.0, 1.0, key="eval_slider", step=0.01, label_visibility="collapsed", on_change=lambda: st.session_state.update(last_evaluation=st.session_state.eval_slider))
        c2.number_input("Eval Input", 0.0, 1.0, key="eval_input", step=0.01, label_visibility="collapsed", on_change=lambda: st.session_state.update(last_evaluation=st.session_state.eval_input))

    # Row 2
    c3, c4, c5 = st.columns(3)
    with c3:
        st.markdown('<div class="feature-card"><span>📅 <strong>Years at Company</strong></span></div>', unsafe_allow_html=True)
        time_spend = st.number_input("Years", 1, 40, 3, label_visibility="collapsed")
    with c4:
        st.markdown('<div class="feature-card"><span>📁 <strong>Number of Projects</strong></span></div>', unsafe_allow_html=True)
        projects = st.number_input("Projects", 1, 10, 4, label_visibility="collapsed")
    with c5:
        st.markdown('<div class="feature-card"><span>⏰ <strong>Avg. Monthly Hours</strong></span></div>', unsafe_allow_html=True)
        hours = st.number_input("Hours", 80, 350, 200, label_visibility="collapsed")

    st.markdown("---")
    _, btn_col, _ = st.columns([1, 2, 1])
    if btn_col.button("🔮 Predict Employee Turnover", use_container_width=True):
        input_data = pd.DataFrame([{
            'satisfaction_level': st.session_state.satisfaction_level,
            'time_spend_company': time_spend,
            'average_monthly_hours': hours,
            'number_project': projects,
            'last_evaluation': st.session_state.last_evaluation
        }])[BEST_FEATURES]
        
        pred = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0] * 100
        
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        r1, r2 = st.columns(2)
        
        with r1:
            cls = "stay-prediction" if pred == 0 else "leave-prediction"
            txt = "STAY" if pred == 0 else "LEAVE"
            msg = "likely to <strong>STAY</strong>" if pred == 0 else "likely to <strong>LEAVE</strong>"
            st.markdown(f'<div class="prediction-box {cls}"><h1>{txt}</h1><p style="font-size: 1.3rem;">Employee is {msg}</p></div>', unsafe_allow_html=True)
        
        with r2:
            st.write(f"**Stay:** {probs[0]:.1f}%")
            st.markdown(f'<div class="progress-bar-container"><div class="progress-bar-green" style="width: {probs[0]}%;"></div></div>', unsafe_allow_html=True)
            st.write(f"**Leave:** {probs[1]:.1f}%")
            st.markdown(f'<div class="progress-bar-container"><div class="progress-bar-red" style="width: {probs[1]}%;"></div></div>', unsafe_allow_html=True)

# ============================================================================
# BATCH PREDICTION TAB
# ============================================================================
def render_batch_prediction_tab(model):
    st.markdown("---")
    st.markdown('<h2 class="section-header">📊 Batch Employee Prediction</h2>', unsafe_allow_html=True)
    
    # Required Columns Info
    with st.expander("📋 View Required Model Features"):
        st.info("The model expects these 5 features. You can map your own columns to these below.")
        c1, c2 = st.columns(2)
        for i, f in enumerate(BEST_FEATURES):
            (c1 if i < 3 else c2).markdown(f"- **{f}**: {FEATURE_DESCRIPTIONS[f]}")

    st.markdown("### 📁 Upload Data")
    c1, c2 = st.columns([1, 2])
    with c1:
        fmt = st.selectbox("File Format", ["CSV", "Excel (.xlsx)"])
    with c2:
        file = st.file_uploader(f"Upload {fmt}", type=["csv"] if fmt == "CSV" else ["xlsx", "xls"])

    st.markdown("### ⚙️ Output Settings")
    c1, c2 = st.columns(2)
    with c1:
        col_name = st.text_input("Prediction Column Name", "Prediction")
    with c2:
        label_opt = st.selectbox("Labels", ["Leave / Stay", "Yes / No", "1 / 0"])
        labels = {1: "Leave", 0: "Stay"} if label_opt == "Leave / Stay" else ({1: "Yes", 0: "No"} if label_opt == "Yes / No" else {1: 1, 0: 0})

    if file:
        st.markdown("---")
        try:
            df = pd.read_csv(file) if fmt == "CSV" else pd.read_excel(file)
            
            # Stats
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="stats-card"><h4>Rows</h4><div class="number">{len(df):,}</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="stats-card"><h4>Columns</h4><div class="number">{len(df.columns)}</div></div>', unsafe_allow_html=True)
            
            missing = [c for c in BEST_FEATURES if c not in df.columns]
            
            # ===== COMPACT COLUMN MAPPING SECTION =====
            st.markdown("### 🔄 Column Mapping")
            
            # Checkbox logic: If missing columns, check it by default
            use_mapping = st.checkbox("Enable Column Mapping", value=bool(missing))
            
            mapping, is_valid = get_compact_column_mapping(df.columns, use_mapping)
            
            # Check validation based on mapping
            final_missing = []
            if use_mapping:
                # If mapping enabled, check if user selection covers requirement
                if not is_valid:
                    final_missing = ["Mapping Error"] # Dummy to block button
            else:
                # If mapping disabled, check exact names
                final_missing = missing

            with c3:
                status_color = "#dc3545" if final_missing else "#28a745"
                status_text = "❌ Missing Cols" if final_missing else "✅ Ready"
                st.markdown(f'<div class="stats-card" style="border-top-color: {status_color};"><h4>Status</h4><div class="number" style="color: {status_color}; font-size: 1.5rem;">{status_text}</div></div>', unsafe_allow_html=True)

            # Data Preview
            st.markdown("#### Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            if final_missing:
                if not use_mapping:
                     st.warning(f"Missing columns: {', '.join(missing)}. Please enable **Column Mapping** above to fix this.")
            else:
                st.markdown("---")
                _, btn_col, _ = st.columns([1, 2, 1])
                
                if btn_col.button("🔮 Generate Batch Predictions", use_container_width=True):
                    with st.spinner("Processing..."):
                        # Create temp dataframe for prediction using mapping
                        pred_df = pd.DataFrame()
                        for feature, user_col in mapping.items():
                            pred_df[feature] = df[user_col]
                        
                        preds = model.predict(pred_df)
                        probs = model.predict_proba(pred_df)
                        
                        # Add to ORIGINAL dataframe
                        res_df = df.copy()
                        res_df[col_name] = [labels[p] for p in preds]
                        res_df[f"{col_name}_Prob_Stay"] = (probs[:, 0] * 100).round(2)
                        res_df[f"{col_name}_Prob_Leave"] = (probs[:, 1] * 100).round(2)
                        
                        st.success("✅ Done!")
                        
                        # Results Summary
                        leavers = sum(preds == 1)
                        pct_leave = (leavers / len(preds)) * 100
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write(f"**Predicted Turnover Rate:** {pct_leave:.1f}%")
                            st.markdown(f'<div class="progress-bar-container"><div class="progress-bar-red" style="width: {pct_leave}%;"></div></div>', unsafe_allow_html=True)
                        
                        with c2:
                            csv = convert_df_to_csv(res_df)
                            st.download_button("📥 Download Results (CSV)", csv, "predictions.csv", "text/csv", use_container_width=True)
                            
                        st.dataframe(res_df.head(), use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    st.markdown('<h1 class="main-header">👥 Employee Turnover Prediction</h1>', unsafe_allow_html=True)
    model = load_model_from_huggingface()
    if model:
        t1, t2 = st.tabs(["📝 Individual Prediction", "📊 Batch Prediction"])
        with t1: render_individual_prediction_tab(model)
        with t2: render_batch_prediction_tab(model)

if __name__ == "__main__":
    main()
