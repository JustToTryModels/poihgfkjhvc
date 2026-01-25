import streamlit as st
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import io
import shap
import matplotlib.pyplot as plt

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
    
    /* Settings card - Light Cream Background (matching feature-card) */
    .settings-card {
        background-color: #FFE8C2;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 5px solid #1E3A5F;
        margin: 1rem 0;
    }
    
    /* Column mapping card - Light Cream Background (matching feature-card) */
    .mapping-card {
        background-color: #FFE8C2;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 5px solid #1E3A5F;
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
    
    /* Download button styling - UPDATED TO FORCE WHITE TEXT */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #28a745, #20c997) !important;
        animation: none !important;
    }
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #20c997, #28a745) !important;
    }
    
    /* Force white text on download buttons for ALL states */
    .stDownloadButton>button,
    .stDownloadButton>button:hover,
    .stDownloadButton>button:active,
    .stDownloadButton>button:focus,
    .stDownloadButton>button:visited,
    .stDownloadButton>button:focus-visible,
    .stDownloadButton>button span,
    .stDownloadButton>button:hover span,
    .stDownloadButton>button:active span {
        color: white !important;
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
    
    /* ===== BEAUTIFUL EXPANDER/DROPDOWN STYLING ===== */
    div[data-testid="stExpander"] {
        border: none !important;
        border-radius: 16px !important;
        margin: 1.5rem 0 !important;
        overflow: visible !important;
    }
    
    div[data-testid="stExpander"] details {
        border: none !important;
        border-radius: 16px !important;
        overflow: visible !important;
    }
    
    /* Closed state - Beautiful gradient header */
    div[data-testid="stExpander"] details summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
        background-size: 200% 200% !important;
        color: white !important;
        border-radius: 16px !important;
        padding: 1.25rem 2rem !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        text-align: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 
            0 8px 25px rgba(102, 126, 234, 0.35),
            0 4px 10px rgba(118, 75, 162, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        animation: expanderGradientFlow 4s ease infinite !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    /* Gradient animation for expander */
    @keyframes expanderGradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Shimmer effect on expander */
    div[data-testid="stExpander"] details summary::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.3),
            transparent
        ) !important;
        animation: expanderShimmer 3s ease-in-out infinite !important;
    }
    
    @keyframes expanderShimmer {
        0% { left: -100%; }
        50% { left: 100%; }
        100% { left: 100%; }
    }
    
    /* Hover state - Lift and glow effect */
    div[data-testid="stExpander"] details summary:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #ec4899 100%) !important;
        background-size: 200% 200% !important;
        transform: translateY(-4px) scale(1.01) !important;
        box-shadow: 
            0 15px 35px rgba(124, 58, 237, 0.4),
            0 8px 20px rgba(168, 85, 247, 0.3),
            0 0 40px rgba(236, 72, 153, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        color: white !important;
    }
    
    /* Arrow/Icon styling */
    div[data-testid="stExpander"] details summary svg {
        color: white !important;
        fill: white !important;
        transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55) !important;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.2)) !important;
    }
    
    /* Icon hover animation */
    div[data-testid="stExpander"] details summary:hover svg {
        transform: scale(1.3) !important;
        filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.3)) !important;
    }
    
    /* Open state - Green success gradient */
    div[data-testid="stExpander"] details[open] summary {
        border-radius: 16px 16px 0 0 !important;
        background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%) !important;
        background-size: 200% 200% !important;
        animation: expanderOpenGlow 3s ease infinite !important;
        box-shadow: 
            0 8px 25px rgba(5, 150, 105, 0.35),
            0 4px 10px rgba(16, 185, 129, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }
    
    @keyframes expanderOpenGlow {
        0% { 
            background-position: 0% 50%;
            box-shadow: 
                0 8px 25px rgba(5, 150, 105, 0.35),
                0 4px 10px rgba(16, 185, 129, 0.25),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
        }
        50% { 
            background-position: 100% 50%;
            box-shadow: 
                0 8px 30px rgba(5, 150, 105, 0.45),
                0 4px 15px rgba(16, 185, 129, 0.35),
                0 0 25px rgba(52, 211, 153, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.3);
        }
        100% { 
            background-position: 0% 50%;
            box-shadow: 
                0 8px 25px rgba(5, 150, 105, 0.35),
                0 4px 10px rgba(16, 185, 129, 0.25),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
        }
    }
    
    /* Icon rotation when open */
    div[data-testid="stExpander"] details[open] summary svg {
        transform: rotate(180deg) scale(1.1) !important;
    }
    
    /* Content area styling */
    div[data-testid="stExpander"] details > div {
        border: 3px solid transparent !important;
        border-image: linear-gradient(135deg, #059669, #10b981, #34d399) 1 !important;
        border-top: none !important;
        border-radius: 0 0 16px 16px !important;
        background: linear-gradient(180deg, #f0fdf4 0%, #ffffff 100%) !important;
        box-shadow: 
            0 10px 30px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(5, 150, 105, 0.1) !important;
        padding: 1.5rem !important;
    }
    
    /* Center the expander summary text */
    div[data-testid="stExpander"] details summary span {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
        text-align: center !important;
    }
    
    div[data-testid="stExpander"] details summary p {
        text-align: center !important;
        width: 100% !important;
        letter-spacing: 0.5px !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.15) !important;
        margin: 0 !important;
    }
    
    /* Checkbox styling */
    .stCheckbox {
        padding: 0.5rem 0;
    }
    .stCheckbox label {
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* SHAP explanation styling */
    .shap-explanation-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .shap-factor-leaving {
        background-color: #fff5f5;
        border-left: 4px solid #dc3545;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .shap-factor-staying {
        background-color: #f0fff4;
        border-left: 4px solid #28a745;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Spacer for SHAP sections */
    .shap-section-spacer {
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Yellow indicator for unmapped columns */
    .needs-mapping-indicator {
        background-color: #fff3cd;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        text-align: center;
        border: 2px solid #ffc107;
        margin-top: 4px;
        font-weight: 600;
        color: #856404;
        animation: pulseYellow 1.5s ease-in-out infinite;
    }
    
    @keyframes pulseYellow {
        0%, 100% {
            background-color: #fff3cd;
            box-shadow: 0 0 0 0 rgba(255, 193, 7, 0.4);
        }
        50% {
            background-color: #ffe69c;
            box-shadow: 0 0 8px 2px rgba(255, 193, 7, 0.6);
        }
    }
    
    /* Mapped column indicator */
    .mapped-indicator {
        background-color: #d4edda;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        text-align: center;
        border: 1px solid #28a745;
        margin-top: 4px;
        font-weight: 500;
        color: #155724;
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
# SHAP FEATURE DESCRIPTIONS FOR BUSINESS INTERPRETATION
# ============================================================================
SHAP_FEATURE_DESCRIPTIONS = {
    'satisfaction_level': {
        'name': 'Job Satisfaction Level',
        'description': 'How satisfied the employee is with their job',
        'high_meaning': 'Employee is very satisfied with their job',
        'low_meaning': 'Employee is dissatisfied with their job',
        'high_effect': 'Satisfied employees are more likely to STAY',
        'low_effect': 'Dissatisfied employees are more likely to LEAVE'
    },
    'number_project': {
        'name': 'Number of Projects',
        'description': 'Number of projects the employee is assigned to',
        'high_meaning': 'Employee handles many projects (high workload)',
        'low_meaning': 'Employee has few projects (possibly underutilized)',
        'high_effect': 'Too many projects can lead to burnout and LEAVING',
        'low_effect': 'Too few projects may indicate disengagement risk'
    },
    'last_evaluation': {
        'name': 'Last Performance Evaluation Score',
        'description': "The employee's most recent performance review score",
        'high_meaning': 'Employee has excellent performance',
        'low_meaning': 'Employee has poor performance',
        'high_effect': 'High performers may leave for better opportunities OR stay due to recognition',
        'low_effect': 'Poor performers may be at risk of leaving or being let go'
    },
    'time_spend_company': {
        'name': 'Years at Company',
        'description': 'How long the employee has worked at the company',
        'high_meaning': 'Long-tenured employee',
        'low_meaning': 'Relatively new employee',
        'high_effect': 'Long-tenure employees may leave if feeling stagnant',
        'low_effect': 'New employees might leave if not properly onboarded'
    },
    'average_monthly_hours': {
        'name': 'Average Monthly Working Hours',
        'description': 'Average number of hours worked per month',
        'high_meaning': 'Employee works long hours (potential burnout)',
        'low_meaning': 'Employee works fewer hours',
        'high_effect': 'Overworked employees are more likely to LEAVE due to burnout',
        'low_effect': 'Employees with reasonable hours are more likely to STAY'
    }
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
# SHAP EXPLAINER (CACHED)
# ============================================================================
@st.cache_resource
def get_shap_explainer(_model):
    """Create and cache SHAP TreeExplainer"""
    try:
        explainer = shap.TreeExplainer(_model)
        return explainer
    except Exception as e:
        st.error(f"❌ Error creating SHAP explainer: {str(e)}")
        return None

# ============================================================================
# SHAP HELPER FUNCTIONS
# ============================================================================
def get_shap_feature_info(feature_name):
    """Get feature description for SHAP interpretation."""
    default = {
        'name': feature_name.replace('_', ' ').title(),
        'description': 'A predictor variable in the model',
        'high_meaning': 'High value of this feature',
        'low_meaning': 'Low value of this feature',
        'high_effect': 'May influence attrition probability',
        'low_effect': 'May influence attrition probability'
    }
    return SHAP_FEATURE_DESCRIPTIONS.get(feature_name, default)

def get_impact_description(shap_value, threshold_high=0.1, threshold_medium=0.05):
    """Convert SHAP value to impact description."""
    abs_shap = abs(shap_value)
    if abs_shap >= threshold_high:
        strength = "STRONG"
    elif abs_shap >= threshold_medium:
        strength = "MODERATE"
    else:
        strength = "WEAK"
    return strength

def generate_shap_explanation(input_df, model, explainer, prediction, prediction_proba):
    """Generate SHAP waterfall plot and detailed explanation for individual prediction."""
    
    # Compute SHAP values
    shap_values_raw = explainer.shap_values(input_df)
    
    # Handle different SHAP versions
    if isinstance(shap_values_raw, list):
        # Old SHAP format (list of arrays)
        shap_values_class_1 = shap_values_raw[1][0]
        expected_value = explainer.expected_value[1]
    else:
        # New SHAP format (3D array)
        shap_values_class_1 = shap_values_raw[0, :, 1]
        expected_value = explainer.expected_value[1]
    
    # Get feature values
    x_sample = input_df.values[0]
    feature_names = BEST_FEATURES
    
    # Create SHAP Explanation object for waterfall plot
    explanation = shap.Explanation(
        values=shap_values_class_1,
        base_values=expected_value,
        data=x_sample,
        feature_names=feature_names
    )
    
    # Generate waterfall plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(explanation, show=False)
    
    pred_class_name = "LEAVE" if prediction == 1 else "STAY"
    prob_leave = prediction_proba[1] * 100
    
    plt.title(
        f"SHAP Waterfall Plot - Explaining Prediction\n"
        f"Predicted: {pred_class_name} | Probability of Leaving: {prob_leave:.1f}%",
        fontsize=12,
        fontweight='bold',
        pad=15
    )
    plt.tight_layout()
    
    # Generate text explanation
    sorted_indices = np.argsort(np.abs(shap_values_class_1))[::-1]
    
    factors_leaving = []
    factors_staying = []
    
    for idx in sorted_indices:
        feat_name = feature_names[idx]
        feat_value = x_sample[idx]
        shap_val = shap_values_class_1[idx]
        info = get_shap_feature_info(feat_name)
        strength = get_impact_description(shap_val)
        
        factor_data = {
            'name': info['name'],
            'value': feat_value,
            'strength': strength,
            'shap_value': shap_val
        }
        
        if shap_val > 0:
            factors_leaving.append(factor_data)
        else:
            factors_staying.append(factor_data)
    
    return fig, factors_leaving, factors_staying

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

# ============================================================================
# INDIVIDUAL PREDICTION TAB
# ============================================================================
def render_individual_prediction_tab(model, explainer):
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
    
    # Store predictions in session state so they persist
    if predict_button:
        input_df = pd.DataFrame([input_data])[BEST_FEATURES]
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        st.session_state['ind_prediction'] = prediction
        st.session_state['ind_proba'] = prediction_proba
        st.session_state['ind_input_df'] = input_df
        st.session_state['ind_submitted'] = True

    # Display results if they exist in session state
    if st.session_state.get('ind_submitted', False):
        prediction = st.session_state['ind_prediction']
        prediction_proba = st.session_state['ind_proba']
        input_df = st.session_state['ind_input_df']
        
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
        
        # ================================================================
        # SHAP EXPLANATION SECTION (DROPDOWN)
        # ================================================================
        st.markdown("---")
        
        with st.expander("**🔍 Why did the model make this prediction? (Click to expand SHAP Explanation)**"):
            
            if explainer is not None:
                try:
                    with st.spinner("Generating SHAP explanation..."):
                        # Generate SHAP explanation
                        fig, factors_leaving, factors_staying = generate_shap_explanation(
                            input_df, model, explainer, prediction, prediction_proba
                        )
                    
                    # Display the waterfall plot
                    st.markdown("### 📊 SHAP Waterfall Plot")
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.markdown("""
                    <p style="text-align: center; color: #666; font-style: italic; margin-top: -10px;">
                        Red bars (positive) → increase probability of leaving | Blue bars (negative) → decrease probability of leaving
                    </p>
                    """, unsafe_allow_html=True)
                    
                    # Display detailed interpretation
                    st.markdown("---")
                    
                    st.markdown("#### 📊 WHY THE MODEL MADE THIS PREDICTION")
                    st.markdown("Here are the main factors that influenced this prediction:")
                    
                    # Factors pushing toward LEAVING
                    if factors_leaving:
                        st.markdown("##### 🔴 FACTORS PUSHING TOWARD LEAVING:")
                        for i, factor in enumerate(factors_leaving, 1):
                            st.markdown(f"""
                            <div class="shap-factor-leaving">
                                <strong>{i}. {factor['name']}</strong><br/>
                                • This employee's value: <strong>{factor['value']:.2f}</strong><br/>
                                • Impact: <strong>{factor['strength']}</strong> push toward LEAVING
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Add spacing between the two sections
                    st.markdown('<div class="shap-section-spacer"></div>', unsafe_allow_html=True)
                    
                    # Factors pushing toward STAYING
                    if factors_staying:
                        st.markdown("##### 🟢 FACTORS PUSHING TOWARD STAYING:")
                        for i, factor in enumerate(factors_staying, 1):
                            st.markdown(f"""
                            <div class="shap-factor-staying">
                                <strong>{i}. {factor['name']}</strong><br/>
                                • This employee's value: <strong>{factor['value']:.2f}</strong><br/>
                                • Impact: <strong>{factor['strength']}</strong> push toward STAYING
                            </div>
                            """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating SHAP explanation: {str(e)}")
                    st.info("The prediction is still valid, but we couldn't generate the detailed explanation.")
            else:
                st.warning("SHAP explainer not available. Unable to generate explanation.")

# ============================================================================
# BATCH PREDICTION TAB
# ============================================================================
def render_batch_prediction_tab(model):
    """Render the batch prediction tab content"""
    
    st.markdown("---")
    st.markdown('<h2 class="section-header">📊 Batch Employee Prediction</h2>', unsafe_allow_html=True)
    
    with st.expander("**📋 Required Columns in Your File (Click to Expand)**"):
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
        include_probabilities = st.toggle("Include prediction probabilities in output", value=True, key="include_probabilities")
        if include_probabilities:
            st.success("✅ Two additional columns will be added: `Probability_Stay` and `Probability_Leave`")
    
    with col2:
        st.markdown("""<div class="settings-card"><h4>⚠️ High Risk Filter</h4></div>""", unsafe_allow_html=True)
        include_high_risk_download = st.toggle("Enable high-risk employees download", value=True, key="include_high_risk")
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
            
            st.markdown("""<div class="settings-card"><h4>🔄 Column Mapping</h4></div>""", unsafe_allow_html=True)
            enable_column_mapping = st.toggle(
                "Enable Column Mapping — Map your columns to model features (temporary, original data unchanged)",
                value=len(missing_columns) > 0,
                key="enable_column_mapping"
            )
            
            column_mapping = {}
            mapping_valid = True
            column_list = list(df.columns)
            
            if enable_column_mapping:
                st.markdown("""
                <div class="mapping-card">
                    <p style="margin: 0; font-size: 0.9rem;">🔧 <strong>Map your columns below.</strong> This is temporary — your original data remains unchanged.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add "Select Column" as placeholder option at the beginning
                mapping_options = ["Select Column"] + column_list
                
                # Create compact mapping table with 5 columns
                cols = st.columns(5)
                for idx, feature in enumerate(BEST_FEATURES):
                    with cols[idx]:
                        st.markdown(f"**{feature.replace('_', ' ').title()}**")
                        
                        # Determine the default index
                        # If the feature exists in the uploaded data columns, auto-select it
                        if feature in column_list:
                            default_index = mapping_options.index(feature)
                        else:
                            default_index = 0  # "Select Column"
                        
                        selected = st.selectbox(
                            f"Map {feature}",
                            options=mapping_options,
                            index=default_index,
                            key=f"map_{feature}",
                            label_visibility="collapsed"
                        )
                        column_mapping[feature] = selected
                        
                        # Add visual indicator based on selection
                        if selected == "Select Column":
                            st.markdown('<div class="needs-mapping-indicator">⚠️ Needs Mapping</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="mapped-indicator">✓ Mapped</div>', unsafe_allow_html=True)
                
                # Validation Logic
                used_columns = list(column_mapping.values())
                
                # Check 1: Ensure no column is left as "Select Column"
                if "Select Column" in used_columns:
                    st.warning("⚠️ Please select a column for every feature to proceed.")
                    mapping_valid = False
                else:
                    # Check 2: Check for duplicates (only if no placeholders exist)
                    duplicate_columns = [col for col in set(used_columns) if used_columns.count(col) > 1]
                    if duplicate_columns:
                        st.error(f"⚠️ Duplicate mapping: **{', '.join(duplicate_columns)}** is mapped to multiple features.")
                        mapping_valid = False
                    else:
                        st.success("✅ Mapping valid! Ready to predict.")
            else:
                # No mapping - use exact column names
                for feature in BEST_FEATURES:
                    column_mapping[feature] = feature
                
                if missing_columns:
                    st.error(f"❌ Missing columns: **{', '.join(missing_columns)}**. Enable Column Mapping above to map your columns.")
                    mapping_valid = False
                else:
                    st.success("✅ All required columns found! Ready to predict.")
            
            # ========================================================================
            # DATA QUALITY CHECKS: NULL VALUES AND NON-NUMERIC DATA
            # ========================================================================
            if mapping_valid:
                # Check for null values in mapped columns
                null_issues = []
                for feature in BEST_FEATURES:
                    source_col = column_mapping[feature]
                    null_count = df[source_col].isnull().sum()
                    if null_count > 0:
                        null_issues.append(f"**{source_col}**: {null_count} null value(s)")
                
                if null_issues:
                    st.warning(f"⚠️ **Null Values Detected:** Your data contains null values in the following columns. Please clean your data before proceeding:\n\n" + "\n\n".join([f"• {issue}" for issue in null_issues]))
                    mapping_valid = False
                
                # Check for non-numeric data types in mapped columns
                non_numeric_issues = []
                for feature in BEST_FEATURES:
                    source_col = column_mapping[feature]
                    if not pd.api.types.is_numeric_dtype(df[source_col]):
                        non_numeric_issues.append(f"**{source_col}** (current type: `{df[source_col].dtype}`)")
                
                if non_numeric_issues:
                    st.warning(f"⚠️ **Non-Numeric Data Detected:** The following columns must contain numeric values (integer or float) for the model to work. Please ensure these columns contain only numbers:\n\n" + "\n\n".join([f"• {issue}" for issue in non_numeric_issues]))
                    mapping_valid = False
            
            # Prediction Section
            if mapping_valid:
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    batch_predict_button = st.button("🔮 Generate Batch Predictions", use_container_width=True, key="batch_predict")
                
                # Logic to process and store batch predictions
                if batch_predict_button:
                    with st.spinner("🔄 Processing predictions..."):
                        # Create temporary dataframe with mapped columns
                        prediction_df = pd.DataFrame()
                        for feature in BEST_FEATURES:
                            prediction_df[feature] = df[column_mapping[feature]].copy()
                        
                        predictions = model.predict(prediction_df)
                        prediction_probabilities = model.predict_proba(prediction_df)
                        
                        # Result dataframe with original columns preserved
                        result_df = df.copy()
                        result_df[prediction_column_name] = [prediction_labels[p] for p in predictions]
                        
                        if include_probabilities:
                            result_df[f"{prediction_column_name}_Probability_Stay"] = (prediction_probabilities[:, 0] * 100).round(2)
                            result_df[f"{prediction_column_name}_Probability_Leave"] = (prediction_probabilities[:, 1] * 100).round(2)
                    
                    # Store everything in session state
                    st.session_state['batch_result_df'] = result_df
                    st.session_state['batch_probabilities'] = prediction_probabilities
                    st.session_state['batch_predictions'] = predictions
                    st.session_state['batch_submitted'] = True

                # Display batch results if they exist in session state
                if st.session_state.get('batch_submitted', False):
                    result_df = st.session_state['batch_result_df']
                    prediction_probabilities = st.session_state['batch_probabilities']
                    predictions = st.session_state['batch_predictions']

                    st.success("✅ Predictions generated successfully!")
                    
                    if enable_column_mapping:
                        st.info("ℹ️ Column mapping was used. Original column names preserved in output.")
                    
                    st.markdown("---")
                    st.markdown("### 📊 Prediction Results")
                    
                    leaving_count = sum(predictions == 1)
                    staying_count = sum(predictions == 0)
                    leaving_percentage = (leaving_count / len(predictions)) * 100
                    staying_percentage = (staying_count / len(predictions)) * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""<div class="stats-card"><h4>Total Employees</h4><div class="number">{len(predictions):,}</div></div>""", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""<div class="stats-card" style="border-top-color: #dc3545;"><h4>Predicted to Leave</h4><div class="number" style="color: #dc3545;">{leaving_count:,}</div><p style="color: #666;">({leaving_percentage:.1f}%)</p></div>""", unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""<div class="stats-card" style="border-top-color: #28a745;"><h4>Predicted to Stay</h4><div class="number" style="color: #28a745;">{staying_count:,}</div><p style="color: #666;">({staying_percentage:.1f}%)</p></div>""", unsafe_allow_html=True)
                    with col4:
                        avg_leave_prob = prediction_probabilities[:, 1].mean() * 100
                        st.markdown(f"""<div class="stats-card" style="border-top-color: #ffc107;"><h4>Avg. Leave Probability</h4><div class="number" style="color: #ffc107;">{avg_leave_prob:.1f}%</div></div>""", unsafe_allow_html=True)
                    
                    st.markdown("#### 📈 Turnover Distribution")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Staying:** {staying_percentage:.1f}%")
                        st.markdown(f"""<div class="progress-bar-container"><div class="progress-bar-green" style="width: {staying_percentage}%;"></div></div>""", unsafe_allow_html=True)
                    with col2:
                        st.write(f"**Leaving:** {leaving_percentage:.1f}%")
                        st.markdown(f"""<div class="progress-bar-container"><div class="progress-bar-red" style="width: {leaving_percentage}%;"></div></div>""", unsafe_allow_html=True)
                    
                    st.markdown("#### 📄 Result Data Preview")
                    st.dataframe(result_df.head(20), use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("### 📥 Download Results")
                    
                    if include_high_risk_download:
                        col1, col2, col3 = st.columns([1, 1, 1])
                    else:
                        col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        csv_data = convert_df_to_csv(result_df)
                        st.download_button(
                            label="📥 Download as CSV", data=csv_data,
                            file_name="employee_predictions.csv", mime="text/csv",
                            use_container_width=True, key="download_csv"
                        )
                    
                    with col2:
                        excel_data = convert_df_to_excel(result_df)
                        st.download_button(
                            label="📥 Download as Excel", data=excel_data,
                            file_name="employee_predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True, key="download_excel"
                        )
                    
                    if include_high_risk_download:
                        with col3:
                            high_risk_df = result_df[prediction_probabilities[:, 1] > 0.5]
                            if len(high_risk_df) > 0:
                                high_risk_csv = convert_df_to_csv(high_risk_df)
                                st.download_button(
                                    label=f"📥 High Risk Only ({len(high_risk_df)})",
                                    data=high_risk_csv, file_name="high_risk_employees.csv",
                                    mime="text/csv", use_container_width=True, key="download_high_risk"
                                )
                            else:
                                st.info("No high-risk employees found")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please ensure your file is properly formatted and not corrupted.")
    
    else:
        st.markdown("---")
        with st.expander("**📋 View Sample Data Format**"):
            sample_data = pd.DataFrame({
                'employee_id': [1, 2, 3, 4, 5],
                'satisfaction_level': [0.38, 0.80, 0.11, 0.72, 0.37],
                'time_spend_company': [3, 5, 4, 3, 2],
                'average_monthly_hours': [157, 262, 272, 223, 159],
                'number_project': [2, 5, 7, 5, 2],
                'last_evaluation': [0.53, 0.86, 0.88, 0.87, 0.52],
                'department': ['sales', 'IT', 'IT', 'sales', 'hr'],
                'salary': ['low', 'medium', 'medium', 'high', 'low']
            })
            st.dataframe(sample_data, use_container_width=True)
            st.info("💡 Only the 5 required columns will be used for prediction. Other columns will be preserved in the output.")

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.markdown('<h1 class="main-header">👥 Employee Turnover Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict whether employees are likely to leave the company</p>', unsafe_allow_html=True)
    
    model = load_model_from_huggingface()
    
    if model is None:
        st.error("❌ Failed to load model. Please check the Hugging Face repository.")
        st.info(f"Repository: https://huggingface.co/{HF_REPO_ID}")
        return
    
    # Create SHAP explainer
    explainer = get_shap_explainer(model)
    
    tab1, tab2 = st.tabs(["📝 Individual Prediction", "📊 Batch Prediction"])
    
    with tab1:
        render_individual_prediction_tab(model, explainer)
    
    with tab2:
        render_batch_prediction_tab(model)

if __name__ == "__main__":
    main()
