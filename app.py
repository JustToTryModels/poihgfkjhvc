import streamlit as st
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import io
import plotly.graph_objects as go
import time

# ============================================================================
# 1. PAGE CONFIGURATION & SETUP
# ============================================================================
st.set_page_config(
    page_title="Employee Insight AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# 2. ADVANCED CSS STYLING (THE "JAW-DROPPING" PART)
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700&display=swap');

    /* --- GLOBAL THEME --- */
    .stApp {
        background-color: #0a0a0a;
        background-image: 
            radial-gradient(circle at 50% 0%, #1a1a2e 0%, transparent 75%),
            radial-gradient(circle at 0% 50%, #1e1e24 0%, transparent 50%),
            radial-gradient(circle at 100% 50%, #0f0f1a 0%, transparent 50%);
        font-family: 'Outfit', sans-serif;
    }
    
    h1, h2, h3, h4, p, div, span, label {
        color: #e0e0e0;
        font-family: 'Outfit', sans-serif !important;
    }

    /* --- ANIMATIONS --- */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 0 0 rgba(124, 58, 237, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(124, 58, 237, 0); }
        100% { box-shadow: 0 0 0 0 rgba(124, 58, 237, 0); }
    }

    /* --- HEADER STYLING --- */
    .hero-container {
        text-align: center;
        padding: 2rem 0 3rem 0;
        animation: fadeIn 1s ease-out;
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #6E85B7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        color: #8b9bb4;
        font-weight: 300;
    }

    /* --- GLASSMORPHISM CARDS --- */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(124, 58, 237, 0.5);
    }

    /* --- CUSTOM TABS --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(255,255,255,0.02);
        padding: 10px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 30px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background-color: transparent;
        border: none !important;
        color: #8b9bb4;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4338ca 0%, #6366f1 100%);
        color: white;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
    }
    
    /* --- INPUT WIDGETS --- */
    .stSlider [data-baseweb="slider-track"] {
        background: linear-gradient(90deg, #4338ca, #d946ef);
    }
    
    .stNumberInput input {
        background-color: rgba(0,0,0,0.2) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
    }

    /* --- BUTTONS --- */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #ec4899, #8b5cf6, #3b82f6);
        background-size: 200% 200%;
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 0.8rem;
        border-radius: 15px;
        border: none;
        box-shadow: 0 10px 25px -5px rgba(139, 92, 246, 0.5);
        transition: all 0.5s ease;
        animation: gradientMove 3s ease infinite;
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 20px 30px -10px rgba(236, 72, 153, 0.6);
    }
    
    .stDownloadButton>button {
        background: rgba(255,255,255,0.05);
        border: 1px solid #3b82f6;
        color: #3b82f6;
    }
    .stDownloadButton>button:hover {
        background: #3b82f6;
        color: white;
    }

    /* --- RESULTS & METRICS --- */
    .result-container-stay {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.2));
        border: 1px solid #10b981;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        animation: fadeIn 0.8s ease-out;
    }
    
    .result-container-leave {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(185, 28, 28, 0.2));
        border: 1px solid #ef4444;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        animation: fadeIn 0.8s ease-out;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#fff, #aaa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #8b9bb4;
        margin-bottom: 5px;
    }
    
    /* Hide standard elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 3. CONSTANTS & CONFIG
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

FEATURE_META = {
    "satisfaction_level": {"icon": "😊", "label": "Satisfaction", "min": 0.0, "max": 1.0, "default": 0.6},
    "time_spend_company": {"icon": "📅", "label": "Years at Company", "min": 1, "max": 20, "default": 3},
    "average_monthly_hours": {"icon": "⏱️", "label": "Monthly Hours", "min": 50, "max": 350, "default": 200},
    "number_project": {"icon": "📂", "label": "Project Count", "min": 1, "max": 10, "default": 4},
    "last_evaluation": {"icon": "📊", "label": "Last Evaluation", "min": 0.0, "max": 1.0, "default": 0.7},
}

# ============================================================================
# 4. UTILITY FUNCTIONS
# ============================================================================
@st.cache_resource
def load_model():
    """Loads model from Hugging Face with spinner."""
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
        return joblib.load(model_path)
    except Exception as e:
        return None

def create_gauge_chart(probability):
    """Creates a beautiful Plotly gauge chart for churn probability."""
    
    color = "#ef4444" if probability > 50 else "#10b981"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability", 'font': {'size': 24, 'color': "#e0e0e0"}},
        number = {'suffix': "%", 'font': {'color': "white"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': "rgba(16, 185, 129, 0.3)"},
                {'range': [50, 100], 'color': "rgba(239, 68, 68, 0.3)"}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 50}}))

    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        font = {'color': "white", 'family': "Outfit"},
        height = 300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# ============================================================================
# 5. MAIN APPLICATION LOGIC
# ============================================================================

def main():
    # --- HEADER ---
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">Employee Insight AI</div>
            <div class="hero-subtitle">Next-Generation Retention Analytics</div>
        </div>
    """, unsafe_allow_html=True)

    # --- MODEL LOADING ---
    with st.spinner("Initializing AI Core..."):
        model = load_model()
        
    if model is None:
        st.error("❌ Critical Error: Neural link to model repository failed.")
        st.info(f"Check repository: {HF_REPO_ID}")
        return

    # --- TABS ---
    tab1, tab2 = st.tabs(["⚡ Individual Analysis", "📂 Batch Processing"])

    # ------------------------------------------------------------------------
    # TAB 1: INDIVIDUAL ANALYSIS
    # ------------------------------------------------------------------------
    with tab1:
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        
        # Two columns layout for inputs
        col_inputs, col_viz = st.columns([1, 1], gap="large")

        with col_inputs:
            st.markdown("### 🧬 Employee Parameters")
            
            # 1. Satisfaction
            st.markdown(f"""
            <div class="glass-card" style="padding: 15px;">
                <div class="metric-label">{FEATURE_META['satisfaction_level']['icon']} {FEATURE_META['satisfaction_level']['label']}</div>
            </div>
            """, unsafe_allow_html=True)
            satisfaction = st.slider("Satisfaction", 0.0, 1.0, 0.5, 0.01, label_visibility="collapsed")
            
            # 2. Evaluation
            st.markdown(f"""
            <div class="glass-card" style="padding: 15px; margin-top: 10px;">
                <div class="metric-label">{FEATURE_META['last_evaluation']['icon']} {FEATURE_META['last_evaluation']['label']}</div>
            </div>
            """, unsafe_allow_html=True)
            evaluation = st.slider("Evaluation", 0.0, 1.0, 0.7, 0.01, label_visibility="collapsed")

            # 3. Numeric Inputs Row
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"<div style='text-align:center; color:#8b9bb4; font-size:0.8rem;'>{FEATURE_META['time_spend_company']['icon']} Years</div>", unsafe_allow_html=True)
                years = st.number_input("Years", 1, 20, 3, label_visibility="collapsed")
            with c2:
                st.markdown(f"<div style='text-align:center; color:#8b9bb4; font-size:0.8rem;'>{FEATURE_META['number_project']['icon']} Projects</div>", unsafe_allow_html=True)
                projects = st.number_input("Projects", 1, 10, 4, label_visibility="collapsed")
            with c3:
                st.markdown(f"<div style='text-align:center; color:#8b9bb4; font-size:0.8rem;'>{FEATURE_META['average_monthly_hours']['icon']} Hours</div>", unsafe_allow_html=True)
                hours = st.number_input("Hours", 50, 400, 200, label_visibility="collapsed")

            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🚀 Run Prediction Model", key="pred_ind")

        # Result Visualization Section
        with col_viz:
            if predict_btn:
                # Prepare data
                input_data = pd.DataFrame([{
                    'satisfaction_level': satisfaction,
                    'time_spend_company': years,
                    'average_monthly_hours': hours,
                    'number_project': projects,
                    'last_evaluation': evaluation
                }])[BEST_FEATURES]

                # Animation delay for effect
                with st.spinner("Analyzing behavioral patterns..."):
                    time.sleep(0.6)
                
                # Predict
                prediction = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0]
                prob_leave = prob[1] * 100
                prob_stay = prob[0] * 100

                # Render Card based on result
                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-container-leave">
                        <div style="font-size: 4rem;">⚠️</div>
                        <h2 style="color: #ef4444; margin: 10px 0;">Risk Detected</h2>
                        <p style="color: #fca5a5;">High probability of turnover identified.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-container-stay">
                        <div style="font-size: 4rem;">🛡️</div>
                        <h2 style="color: #10b981; margin: 10px 0;">Stable</h2>
                        <p style="color: #6ee7b7;">Employee likely to remain with organization.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Plotly Gauge
                st.plotly_chart(create_gauge_chart(prob_leave), use_container_width=True)
                
            else:
                # Placeholder state
                st.markdown("""
                <div class="glass-card" style="height: 100%; display: flex; align-items: center; justify-content: center; text-align: center; flex-direction: column; min-height: 400px;">
                    <div style="font-size: 3rem; margin-bottom: 20px; opacity: 0.5;">🔮</div>
                    <h3 style="color: #8b9bb4;">Awaiting Input</h3>
                    <p style="color: #555;">Adjust parameters on the left to generate prediction.</p>
                </div>
                """, unsafe_allow_html=True)

    # ------------------------------------------------------------------------
    # TAB 2: BATCH PROCESSING
    # ------------------------------------------------------------------------
    with tab2:
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        
        # File Upload Area
        st.markdown("""
        <div class="glass-card">
            <h3>📂 Data Ingestion</h3>
            <p style="color: #8b9bb4; font-size: 0.9rem;">Supports CSV or Excel. Required columns: <code>satisfaction_level</code>, <code>time_spend_company</code>, <code>average_monthly_hours</code>, <code>number_project</code>, <code>last_evaluation</code>.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=["csv", "xlsx"], label_visibility="collapsed")

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                # Validate columns
                missing_cols = [col for col in BEST_FEATURES if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Schema Mismatch. Missing columns: {', '.join(missing_cols)}")
                else:
                    st.success("✅ Schema Validated. Ready for processing.")
                    
                    if st.button("⚡ Process Batch", key="batch_proc"):
                        with st.status("Processing Data Pipeline...", expanded=True) as status:
                            st.write("Extracting features...")
                            X = df[BEST_FEATURES]
                            time.sleep(0.5)
                            
                            st.write("Running inference engine...")
                            preds = model.predict(X)
                            probs = model.predict_proba(X)
                            time.sleep(0.5)
                            
                            st.write("Compiling results...")
                            df['Prediction'] = preds
                            df['Risk_Label'] = df['Prediction'].map({1: 'Leaving', 0: 'Staying'})
                            df['Leave_Probability'] = (probs[:, 1] * 100).round(2)
                            status.update(label="Processing Complete!", state="complete", expanded=False)

                        # Results Dashboard
                        st.markdown("---")
                        
                        # Metrics Row
                        total = len(df)
                        at_risk = df['Prediction'].sum()
                        risk_pct = (at_risk / total) * 100
                        
                        m1, m2, m3 = st.columns(3)
                        
                        with m1:
                            st.markdown(f"""
                            <div class="glass-card" style="text-align: center;">
                                <div class="metric-label">Total Records</div>
                                <div class="metric-value">{total:,}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with m2:
                            st.markdown(f"""
                            <div class="glass-card" style="text-align: center; border-color: #ef4444;">
                                <div class="metric-label" style="color: #fca5a5;">At Risk</div>
                                <div class="metric-value" style="background: -webkit-linear-gradient(#ef4444, #f87171); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{at_risk:,}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with m3:
                            st.markdown(f"""
                            <div class="glass-card" style="text-align: center; border-color: #3b82f6;">
                                <div class="metric-label" style="color: #93c5fd;">Risk Rate</div>
                                <div class="metric-value" style="background: -webkit-linear-gradient(#3b82f6, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{risk_pct:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Data Preview
                        st.markdown("### 📋 Analysis Report")
                        st.dataframe(
                            df.style.apply(
                                lambda x: ['background-color: rgba(239, 68, 68, 0.2)' if v == 'Leaving' else '' for v in x], 
                                subset=['Risk_Label']
                            ),
                            use_container_width=True,
                            height=300
                        )

                        # Downloads
                        col_d1, col_d2 = st.columns(2)
                        with col_d1:
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Download CSV Report",
                                data=csv,
                                file_name="turnover_prediction_report.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        with col_d2:
                            # Excel handling
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                df.to_excel(writer, index=False, sheet_name='Predictions')
                            st.download_button(
                                "📥 Download Excel Report",
                                data=buffer.getvalue(),
                                file_name="turnover_prediction_report.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )

            except Exception as e:
                st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
