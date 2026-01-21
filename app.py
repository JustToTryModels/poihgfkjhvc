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
# STUNNING CSS STYLING - JAW-DROPPING DESIGN
# ============================================================================
st.markdown("""
<style>
    /* ===== IMPORT STUNNING FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&family=Orbitron:wght@400;700;900&display=swap');
    
    /* ===== GLOBAL BODY STYLING WITH ANIMATED BACKGROUND ===== */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        min-height: 100vh;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        25% { background-position: 100% 50%; }
        50% { background-position: 100% 100%; }
        75% { background-position: 0% 100%; }
        100% { background-position: 0% 50%; }
    }
    
    /* ===== FLOATING PARTICLES BACKGROUND ===== */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(255, 255, 255, 0.1) 2px, transparent 2px),
            radial-gradient(circle at 80% 20%, rgba(255, 255, 255, 0.15) 1px, transparent 1px),
            radial-gradient(circle at 40% 40%, rgba(255, 255, 255, 0.08) 3px, transparent 3px);
        background-size: 100px 100px, 150px 150px, 200px 200px;
        animation: float 20s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        33% { transform: translateY(-20px) rotate(120deg); }
        66% { transform: translateY(-40px) rotate(240deg); }
    }
    
    /* ===== MAIN CONTAINER WITH GLASS MORPHISM ===== */
    .block-container {
        max-width: 1400px !important;
        padding: 3rem 2rem !important;
        margin: 0 auto !important;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 30px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 
            0 25px 45px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        margin-top: 2rem !important;
        margin-bottom: 2rem !important;
        animation: slideInUp 1s ease-out;
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* ===== SPECTACULAR HEADERS WITH GLOWING TEXT ===== */
    .main-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
        animation: titleGlow 3s ease-in-out infinite alternate;
        position: relative;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 200px;
        height: 4px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        border-radius: 2px;
        animation: lineGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes titleGlow {
        0% { text-shadow: 0 0 30px rgba(102, 126, 234, 0.5); }
        100% { text-shadow: 0 0 50px rgba(118, 75, 162, 0.8); }
    }
    
    @keyframes lineGlow {
        0% { box-shadow: 0 0 10px rgba(102, 126, 234, 0.6); }
        100% { box-shadow: 0 0 25px rgba(118, 75, 162, 1); }
    }
    
    .sub-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.4rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 500;
        opacity: 0.9;
    }
    
    .section-header {
        font-family: 'Poppins', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
    }
    
    /* ===== ABSOLUTELY STUNNING TAB DESIGN ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 30px;
        justify-content: center;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 25px;
        padding: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 100px !important;
        min-width: 350px !important;
        padding: 0 40px !important;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.1));
        backdrop-filter: blur(20px);
        border-radius: 20px !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        color: #2c3e50 !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-transform: none !important;
        letter-spacing: 1px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stTabs [data-baseweb="tab"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        transition: left 0.5s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover::before {
        left: 100%;
    }
    
    .stTabs [data-baseweb="tab"] p {
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        color: inherit !important;
        margin: 0 !important;
        padding: 0 !important;
        z-index: 2 !important;
        position: relative !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.15));
        transform: translateY(-8px) scale(1.02);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.15),
            0 0 30px rgba(102, 126, 234, 0.3);
        border: 2px solid rgba(102, 126, 234, 0.5) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: 2px solid #667eea !important;
        box-shadow: 
            0 25px 50px rgba(102, 126, 234, 0.4),
            0 0 40px rgba(118, 75, 162, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        transform: translateY(-5px) scale(1.05);
    }
    
    .stTabs [aria-selected="true"] p {
        color: white !important;
        font-weight: 800 !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 30px;
    }
    
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    
    /* ===== GORGEOUS FEATURE CARDS WITH PREMIUM DESIGN ===== */
    .feature-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.25), rgba(255, 255, 255, 0.1));
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.3),
            0 0 0 1px rgba(255, 255, 255, 0.2);
        border: none;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        background-size: 200% 100%;
        animation: borderFlow 3s ease-in-out infinite;
    }
    
    @keyframes borderFlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .feature-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 
            0 30px 60px rgba(0, 0, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.4),
            0 0 40px rgba(102, 126, 234, 0.2);
    }
    
    .feature-card span {
        font-family: 'Poppins', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        text-shadow: 0 1px 2px rgba(255, 255, 255, 0.8);
    }
    
    /* ===== SPECTACULAR PREDICTION BOXES ===== */
    .prediction-box {
        padding: 3rem;
        border-radius: 25px;
        text-align: center;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
        animation: float 6s ease-in-out infinite;
    }
    
    .stay-prediction {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.9), rgba(34, 197, 94, 0.8));
        border: 2px solid rgba(40, 167, 69, 0.5);
        color: white;
    }
    
    .leave-prediction {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.9), rgba(239, 68, 68, 0.8));
        border: 2px solid rgba(220, 53, 69, 0.5);
        color: white;
    }
    
    .prediction-box h1 {
        font-family: 'Orbitron', sans-serif;
        font-size: 3rem;
        font-weight: 900;
        margin-bottom: 1rem;
        text-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        animation: pulse 2s ease-in-out infinite;
    }
    
    .prediction-box p {
        font-family: 'Poppins', sans-serif;
        font-size: 1.3rem;
        font-weight: 500;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* ===== AMAZING STATS CARDS ===== */
    .stats-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.25), rgba(255, 255, 255, 0.1));
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stats-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.15);
    }
    
    .stats-card h4 {
        font-family: 'Poppins', sans-serif;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .stats-card .number {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        font-weight: 900;
        color: #667eea;
        text-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
        margin-bottom: 0.5rem;
    }
    
    /* ===== MIND-BLOWING PROGRESS BARS ===== */
    .progress-bar-container {
        width: 100%;
        background: rgba(233, 236, 239, 0.3);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        margin: 10px 0 20px 0;
        height: 30px;
        overflow: hidden;
        position: relative;
        box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    .progress-bar-green {
        height: 100%;
        background: linear-gradient(90deg, #28a745, #20c997, #28a745);
        background-size: 200% 100%;
        border-radius: 15px;
        transition: width 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        animation: progressFlow 2s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    .progress-bar-red {
        height: 100%;
        background: linear-gradient(90deg, #dc3545, #e74c3c, #dc3545);
        background-size: 200% 100%;
        border-radius: 15px;
        transition: width 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        animation: progressFlow 2s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    @keyframes progressFlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .progress-bar-green::after,
    .progress-bar-red::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        animation: progressShine 1.5s ease-in-out infinite;
    }
    
    @keyframes progressShine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* ===== SPECTACULAR RAINBOW PREDICT BUTTON ===== */
    .stButton>button {
        width: 100%;
        background: linear-gradient(
            45deg, 
            #ff0080, #ff8c00, #40e0d0, #ff0080, #ff8c00
        );
        background-size: 400% 400%;
        color: white !important;
        font-family: 'Orbitron', sans-serif;
        font-size: 1.5rem;
        font-weight: 900 !important;
        padding: 1.5rem 3rem;
        border-radius: 50px;
        border: none !important;
        outline: none !important;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 15px 35px rgba(255, 0, 128, 0.4),
            0 20px 60px rgba(255, 140, 0, 0.3),
            0 0 80px rgba(64, 224, 208, 0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: buttonGradient 3s ease infinite, buttonFloat 4s ease-in-out infinite;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
    }
    
    @keyframes buttonGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes buttonFloat {
        0%, 100% { transform: translateY(0px) scale(1); }
        50% { transform: translateY(-5px) scale(1.02); }
    }
    
    .stButton>button:hover {
        background: linear-gradient(
            45deg, 
            #00f5ff, #ff00ff, #ffff00, #00f5ff, #ff00ff
        );
        background-size: 400% 400%;
        transform: translateY(-10px) scale(1.05);
        box-shadow: 
            0 25px 50px rgba(0, 245, 255, 0.5),
            0 30px 80px rgba(255, 0, 255, 0.4),
            0 0 150px rgba(255, 255, 0, 0.3),
            inset 0 0 30px rgba(255, 255, 255, 0.1);
        animation: buttonGradient 1.5s ease infinite;
    }
    
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
            rgba(255, 255, 255, 0.6),
            transparent
        );
        transition: left 0.7s ease;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button::after {
        content: '✨';
        position: absolute;
        font-size: 1.5rem;
        right: 25px;
        animation: sparkle 1.5s ease-in-out infinite;
    }
    
    /* ===== STUNNING INPUT FIELD STYLING ===== */
    .stNumberInput > div > div > input,
    .stSlider > div > div > div > div {
        background: rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(15px) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 15px !important;
        color: #2c3e50 !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSlider > div > div > div > div:hover {
        border: 2px solid #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* ===== GORGEOUS SLIDER STYLING ===== */
    .stSlider > div > div > div {
        background: rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(15px) !important;
        border-radius: 15px !important;
    }
    
    /* ===== BEAUTIFUL SELECTBOX STYLING ===== */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(15px) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 15px !important;
    }
    
    /* ===== ELEGANT EXPANDER STYLING ===== */
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    div[data-testid="stExpander"] details summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 15px !important;
        padding: 1rem 1.5rem !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    /* ===== BEAUTIFUL BOX STYLING ===== */
    .success-box, .error-box, .info-box, .required-cols-box {
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    .success-box {
        background: rgba(212, 237, 218, 0.8);
        border-color: rgba(40, 167, 69, 0.5);
    }
    
    .error-box {
        background: rgba(248, 215, 218, 0.8);
        border-color: rgba(220, 53, 69, 0.5);
    }
    
    .info-box, .required-cols-box {
        background: rgba(255, 243, 205, 0.8);
        border-color: rgba(255, 193, 7, 0.5);
    }
    
    /* ===== UPLOAD SECTION STYLING ===== */
    .upload-section {
        background: rgba(248, 249, 250, 0.2);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2.5rem;
        border: 2px dashed rgba(102, 126, 234, 0.5);
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: rgba(102, 126, 234, 0.8);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.2);
    }
    
    /* ===== SETTINGS CARD STYLING ===== */
    .settings-card {
        background: rgba(232, 244, 248, 0.3);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(23, 162, 184, 0.3);
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    .settings-card h4 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    /* ===== DOWNLOAD BUTTON STYLING ===== */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #28a745, #20c997) !important;
        color: white !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 15px !important;
        padding: 0.8rem 1.5rem !important;
        border: none !important;
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #20c997, #28a745) !important;
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(32, 201, 151, 0.4) !important;
    }
    
    /* ===== CHECKBOX STYLING ===== */
    .stCheckbox {
        padding: 1rem 0;
    }
    
    .stCheckbox label {
        font-family: 'Poppins', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        color: #2c3e50;
    }
    
    /* ===== DATAFRAME STYLING ===== */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
    }
    
    /* ===== HIDE STREAMLIT BRANDING ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ===== RESPONSIVE DESIGN ===== */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            min-width: 250px !important;
            font-size: 1.1rem !important;
        }
        
        .feature-card {
            padding: 1.5rem;
        }
        
        .prediction-box {
            padding: 2rem;
        }
        
        .prediction-box h1 {
            font-size: 2rem;
        }
        
        .block-container {
            padding: 2rem 1rem !important;
            margin-top: 1rem !important;
        }
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
    """Sync satisfaction level from slider to session state"""
    st.session_state.satisfaction_level = st.session_state.sat_slider

def sync_satisfaction_input():
    """Sync satisfaction level from number input to session state"""
    st.session_state.satisfaction_level = st.session_state.sat_input

def sync_evaluation_slider():
    """Sync evaluation from slider to session state"""
    st.session_state.last_evaluation = st.session_state.eval_slider

def sync_evaluation_input():
    """Sync evaluation from number input to session state"""
    st.session_state.last_evaluation = st.session_state.eval_input

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def convert_df_to_csv(df):
    """Convert dataframe to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

def convert_df_to_excel(df):
    """Convert dataframe to Excel for download"""
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
    st.markdown('<h2 class="section-header">🎯 Enter Employee Information</h2>', unsafe_allow_html=True)
    
    # Initialize session state for syncing slider and number input
    if 'satisfaction_level' not in st.session_state:
        st.session_state.satisfaction_level = 0.5
    if 'last_evaluation' not in st.session_state:
        st.session_state.last_evaluation = 0.7
    
    # ========================================================================
    # ROW 1: Satisfaction Level & Last Evaluation (side by side with feature cards)
    # ========================================================================
    with st.container():
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <span style="font-size: 1.4rem;">😊 <strong>Satisfaction Level</strong></span>
                <p style="margin-top: 0.5rem; color: #6c757d; font-size: 0.9rem;">Rate employee happiness and job satisfaction</p>
            </div>
            """, unsafe_allow_html=True)
            
            sat_col1, sat_col2 = st.columns([3, 1])
            with sat_col1:
                st.slider(
                    "Satisfaction Slider",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.satisfaction_level,
                    step=0.01,
                    help="Employee satisfaction level (0 = Very Dissatisfied, 1 = Very Satisfied)",
                    label_visibility="collapsed",
                    key="sat_slider",
                    on_change=sync_satisfaction_slider
                )
            with sat_col2:
                st.number_input(
                    "Satisfaction Input",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.satisfaction_level,
                    step=0.01,
                    format="%.2f",
                    label_visibility="collapsed",
                    key="sat_input",
                    on_change=sync_satisfaction_input
                )
            
            satisfaction_level = st.session_state.satisfaction_level
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <span style="font-size: 1.4rem;">📊 <strong>Performance Evaluation</strong></span>
                <p style="margin-top: 0.5rem; color: #6c757d; font-size: 0.9rem;">Latest performance review score</p>
            </div>
            """, unsafe_allow_html=True)
            
            eval_col1, eval_col2 = st.columns([3, 1])
            with eval_col1:
                st.slider(
                    "Evaluation Slider",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.last_evaluation,
                    step=0.01,
                    help="Last performance evaluation score (0 = Poor, 1 = Excellent)",
                    label_visibility="collapsed",
                    key="eval_slider",
                    on_change=sync_evaluation_slider
                )
            with eval_col2:
                st.number_input(
                    "Evaluation Input",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.last_evaluation,
                    step=0.01,
                    format="%.2f",
                    label_visibility="collapsed",
                    key="eval_input",
                    on_change=sync_evaluation_input
                )
            
            last_evaluation = st.session_state.last_evaluation
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========================================================================
    # ROW 2: Years at Company, Number of Projects, Average Monthly Hours (3 columns)
    # ========================================================================
    col3, col4, col5 = st.columns(3, gap="large")
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <span style="font-size: 1.4rem;">📅 <strong>Years at Company</strong></span>
            <p style="margin-top: 0.5rem; color: #6c757d; font-size: 0.9rem;">Total years of service</p>
        </div>
        """, unsafe_allow_html=True)
        time_spend_company = st.number_input(
            "Years",
            min_value=1,
            max_value=40,
            value=3,
            step=1,
            label_visibility="collapsed",
            help="Number of years the employee has worked at the company",
            key="individual_years"
        )
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <span style="font-size: 1.4rem;">📁 <strong>Active Projects</strong></span>
            <p style="margin-top: 0.5rem; color: #6c757d; font-size: 0.9rem;">Current project workload</p>
        </div>
        """, unsafe_allow_html=True)
        number_project = st.number_input(
            "Projects",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            label_visibility="collapsed",
            help="Number of projects the employee is currently working on",
            key="individual_projects"
        )
    
    with col5:
        st.markdown("""
        <div class="feature-card">
            <span style="font-size: 1.4rem;">⏰ <strong>Monthly Hours</strong></span>
            <p style="margin-top: 0.5rem; color: #6c757d; font-size: 0.9rem;">Average work hours per month</p>
        </div>
        """, unsafe_allow_html=True)
        average_monthly_hours = st.number_input(
            "Hours",
            min_value=80,
            max_value=350,
            value=200,
            step=5,
            label_visibility="collapsed",
            help="Average number of hours worked per month",
            key="individual_hours"
        )
    
    # Create input dictionary
    input_data = {
        'satisfaction_level': satisfaction_level,
        'time_spend_company': time_spend_company,
        'average_monthly_hours': average_monthly_hours,
        'number_project': number_project,
        'last_evaluation': last_evaluation
    }
    
    # ========================================================================
    # PREDICTION BUTTON
    # ========================================================================
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Employee Turnover", use_container_width=True, key="individual_predict")
    
    # ========================================================================
    # PREDICTION RESULTS
    # ========================================================================
    if predict_button:
        # Create input DataFrame with correct feature order
        input_df = pd.DataFrame([input_data])[BEST_FEATURES]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        prob_stay = prediction_proba[0] * 100
        prob_leave = prediction_proba[1] * 100
        
        st.markdown("---")
        st.markdown('<h2 class="section-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
        
        # Results in two columns
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            if prediction == 0:
                st.markdown("""
                <div class="prediction-box stay-prediction">
                    <h1>✅ WILL STAY</h1>
                    <p style="font-size: 1.4rem; margin-top: 1.5rem;">
                        Employee is <strong>LIKELY TO STAY</strong> with the company
                    </p>
                    <div style="font-size: 3rem; margin-top: 1rem;">🏢💚</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-box leave-prediction">
                    <h1>⚠️ WILL LEAVE</h1>
                    <p style="font-size: 1.4rem; margin-top: 1.5rem;">
                        Employee is <strong>LIKELY TO LEAVE</strong> the company
                    </p>
                    <div style="font-size: 3rem; margin-top: 1rem;">🚪💔</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Confidence Levels")
            
            # Stay probability with GREEN bar
            st.markdown(f"**🟢 Probability of Staying: {prob_stay:.1f}%**")
            st.markdown(f"""
            <div class="progress-bar-container">
                <div class="progress-bar-green" style="width: {prob_stay}%;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Leave probability with RED bar
            st.markdown(f"**🔴 Probability of Leaving: {prob_leave:.1f}%**")
            st.markdown(f"""
            <div class="progress-bar-container">
                <div class="progress-bar-red" style="width: {prob_leave}%;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk level indicator
            if prob_leave > 70:
                risk_level = "🔴 HIGH RISK"
                risk_color = "#dc3545"
            elif prob_leave > 40:
                risk_level = "🟡 MODERATE RISK"
                risk_color = "#ffc107"
            else:
                risk_level = "🟢 LOW RISK"
                risk_color = "#28a745"
            
            st.markdown(f"""
            <div style="
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(15px);
                border-radius: 15px;
                padding: 1rem;
                text-align: center;
                margin-top: 1rem;
                border: 2px solid {risk_color};
                color: {risk_color};
                font-weight: bold;
                font-size: 1.2rem;
            ">
                {risk_level}
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# BATCH PREDICTION TAB
# ============================================================================
def render_batch_prediction_tab(model):
    """Render the batch prediction tab content"""
    
    st.markdown("---")
    st.markdown('<h2 class="section-header">📊 Batch Employee Analysis</h2>', unsafe_allow_html=True)
    
    # ========================================================================
    # REQUIRED COLUMNS INFO - NOW AS DROPDOWN/EXPANDER
    # ========================================================================
    with st.expander("📋 Required Data Columns (Click to View Details)"):
        st.markdown("""
        <div style="background: rgba(255, 243, 205, 0.8); backdrop-filter: blur(15px); padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem; border: 1px solid rgba(255, 193, 7, 0.5);">
            <h4 style="color: #856404; margin-bottom: 1rem;">📌 Essential Information</h4>
            <p>Your uploaded file <strong>must contain</strong> these columns with <strong>exact names</strong>:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display required features in a beautiful format
        col1, col2 = st.columns(2)
        with col1:
            for i, feature in enumerate(BEST_FEATURES[:3]):
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
                    backdrop-filter: blur(10px);
                    padding: 1.2rem; 
                    border-radius: 12px; 
                    margin: 0.8rem 0; 
                    border: 1px solid rgba(102, 126, 234, 0.2);
                    transition: all 0.3s ease;
                ">
                    <div style="font-size: 1.1rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">
                        {i+1}. {feature}
                    </div>
                    <div style="font-size: 0.9rem; color: #6c757d;">
                        {FEATURE_DESCRIPTIONS[feature]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        with col2:
            for i, feature in enumerate(BEST_FEATURES[3:], 4):
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
                    backdrop-filter: blur(10px);
                    padding: 1.2rem; 
                    border-radius: 12px; 
                    margin: 0.8rem 0; 
                    border: 1px solid rgba(102, 126, 234, 0.2);
                    transition: all 0.3s ease;
                ">
                    <div style="font-size: 1.1rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">
                        {i}. {feature}
                    </div>
                    <div style="font-size: 0.9rem; color: #6c757d;">
                        {FEATURE_DESCRIPTIONS[feature]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.info("💡 **Pro Tip:** Your file can contain additional columns (employee_id, department, etc.). They'll be preserved in output!")
    
    st.markdown("---")
    
    # ========================================================================
    # FILE UPLOAD SECTION
    # ========================================================================
    st.markdown('<h3 style="font-family: Poppins; color: #2c3e50; font-weight: 600;">📁 Upload Your Data File</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown("""
        <div class="settings-card">
            <h4>⚙️ File Configuration</h4>
            <p style="color: #6c757d; font-size: 0.9rem;">Choose your preferred file format</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File format selection
        file_format = st.selectbox(
            "📄 Select file format",
            options=["CSV", "Excel (.xlsx)"],
            index=0,
            help="Choose the format of your data file",
            key="file_format"
        )
    
    with col2:
        st.markdown("""
        <div class="upload-section">
            <h4 style="text-align: center; color: #2c3e50; margin-bottom: 1rem;">🎯 Drop Your File Here</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader based on format
        if file_format == "CSV":
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=["csv"],
                help="Upload a CSV file containing employee data",
                key="csv_uploader",
                label_visibility="collapsed"
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload Excel file",
                type=["xlsx", "xls"],
                help="Upload an Excel file containing employee data",
                key="excel_uploader",
                label_visibility="collapsed"
            )
    
    st.markdown("---")
    
    # ========================================================================
    # PREDICTION OUTPUT SETTINGS
    # ========================================================================
    st.markdown('<h3 style="font-family: Poppins; color: #2c3e50; font-weight: 600;">⚙️ Customize Your Output</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="settings-card">
            <h4>📝 Prediction Column Settings</h4>
            <p style="color: #6c757d; font-size: 0.9rem;">Customize how predictions appear in your results</p>
        </div>
        """, unsafe_allow_html=True)
        
        column_name_option = st.selectbox(
            "🏷️ Choose prediction column name",
            options=["Prediction", "Churn", "Will_Leave", "Turnover", "Custom"],
            index=0,
            help="Select the column name where predictions will be stored",
            key="column_name_option"
        )
        
        if column_name_option == "Custom":
            custom_column_name = st.text_input(
                "✏️ Enter custom name",
                value="My_Prediction",
                help="Enter your preferred column name",
                key="custom_column_name"
            )
            prediction_column_name = custom_column_name if custom_column_name.strip() else "Prediction"
        else:
            prediction_column_name = column_name_option
    
    with col2:
        st.markdown("""
        <div class="settings-card">
            <h4>🏷️ Result Labels</h4>
            <p style="color: #6c757d; font-size: 0.9rem;">Choose how to display the prediction outcomes</p>
        </div>
        """, unsafe_allow_html=True)
        
        label_option = st.selectbox(
            "🎯 Select prediction labels",
            options=["Leave / Stay", "Yes / No", "Churn / Not Churn", "1 / 0", "True / False", "Custom"],
            index=0,
            help="Choose how predictions should be labeled",
            key="label_option"
        )
        
        # Define label mappings (1 = leaving, 0 = staying)
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
                custom_leave_label = st.text_input(
                    "🚪 Label for LEAVING",
                    value="Leaving",
                    help="Label when employee is predicted to leave",
                    key="custom_leave_label"
                )
            with custom_col2:
                custom_stay_label = st.text_input(
                    "🏢 Label for STAYING",
                    value="Staying",
                    help="Label when employee is predicted to stay",
                    key="custom_stay_label"
                )
            prediction_labels = {
                1: custom_leave_label if custom_leave_label.strip() else "Leaving",
                0: custom_stay_label if custom_stay_label.strip() else "Staying"
            }
        else:
            prediction_labels = label_mappings[label_option]
        
        st.success(f"🔍 **Preview:** Leaving → `{prediction_labels[1]}` | Staying → `{prediction_labels[0]}`")
    
    # ========================================================================
    # ADDITIONAL OUTPUT OPTIONS
    # ========================================================================
    st.markdown("---")
    st.markdown('<h3 style="font-family: Poppins; color: #2c3e50; font-weight: 600;">📊 Advanced Output Features</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="settings-card">
            <h4>🎯 Probability Analysis</h4>
            <p style="color: #6c757d; font-size: 0.9rem;">Add detailed confidence scores to your results</p>
        </div>
        """, unsafe_allow_html=True)
        
        include_probabilities = st.checkbox(
            "📈 Include prediction probabilities",
            value=True,
            help="Add columns showing the probability (%) of staying and leaving",
            key="include_probabilities"
        )
        
        if include_probabilities:
            st.success("✅ Two probability columns will be added: `Probability_Stay` and `Probability_Leave`")
        else:
            st.info("ℹ️ Only the main prediction column will be included")
    
    with col2:
        st.markdown("""
        <div class="settings-card">
            <h4>⚠️ High-Risk Employee Filter</h4>
            <p style="color: #6c757d; font-size: 0.9rem;">Separate download for employees at risk of leaving</p>
        </div>
        """, unsafe_allow_html=True)
        
        include_high_risk_download = st.checkbox(
            "🚨 Enable high-risk employee filter",
            value=True,
            help="Separate download for employees with >50% probability of leaving",
            key="include_high_risk"
        )
        
        if include_high_risk_download:
            st.success("✅ High-risk employees can be downloaded separately")
        else:
            st.info("ℹ️ Only complete results will be available for download")
    
    # ========================================================================
    # PROCESS UPLOADED FILE
    # ========================================================================
    if uploaded_file is not None:
        st.markdown("---")
        st.markdown('<h3 style="font-family: Poppins; color: #2c3e50; font-weight: 600;">📄 Data Analysis Preview</h3>', unsafe_allow_html=True)
        
        try:
            # Read the file based on format
            if file_format == "CSV":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Display file info in beautiful cards
            col1, col2, col3 = st.columns(3, gap="medium")
            with col1:
                st.markdown(f"""
                <div class="stats-card">
                    <h4>📊 Total Records</h4>
                    <div class="number">{len(df):,}</div>
                    <p style="color: #6c757d; font-size: 0.9rem;">Employee entries</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="stats-card">
                    <h4>📋 Total Columns</h4>
                    <div class="number">{len(df.columns)}</div>
                    <p style="color: #6c757d; font-size: 0.9rem;">Data fields</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                available_features = [col for col in BEST_FEATURES if col in df.columns]
                st.markdown(f"""
                <div class="stats-card">
                    <h4>✅ Required Columns</h4>
                    <div class="number" style="color: {'#28a745' if len(available_features) == 5 else '#dc3545'};">{len(available_features)}/5</div>
                    <p style="color: #6c757d; font-size: 0.9rem;">Found in file</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show data preview
            st.markdown("#### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Check for missing required columns
            missing_columns = [col for col in BEST_FEATURES if col not in df.columns]
            
            if missing_columns:
                st.markdown(f"""
                <div class="error-box">
                    <h4>❌ Missing Required Columns</h4>
                    <p>Your file is missing the following essential columns:</p>
                    <div style="margin: 1rem 0;">
                        {''.join([f'<div style="background: rgba(220, 53, 69, 0.1); padding: 0.5rem; margin: 0.3rem 0; border-radius: 8px; border-left: 4px solid #dc3545;"><strong>{col}</strong><br><small style="color: #6c757d;">{FEATURE_DESCRIPTIONS[col]}</small></div>' for col in missing_columns])}
                    </div>
                    <p>Please ensure your file contains all required columns with exact names.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                    <h4>🎉 Perfect! All Required Columns Found</h4>
                    <p>Your file contains all necessary columns for prediction. Ready to generate insights!</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show which columns will be used
                with st.expander("🔍 Preview Prediction Columns"):
                    for i, feature in enumerate(BEST_FEATURES, 1):
                        sample_values = df[feature].head(3).tolist()
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, rgba(40, 167, 69, 0.1), rgba(34, 197, 94, 0.1));
                            border-radius: 8px;
                            padding: 1rem;
                            margin: 0.5rem 0;
                            border-left: 4px solid #28a745;
                        ">
                            <strong>{i}. {feature}</strong><br>
                            <small style="color: #6c757d;">Sample values: {sample_values}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Prediction button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    batch_predict_button = st.button(
                        "🚀 Generate Predictions",
                        use_container_width=True,
                        key="batch_predict"
                    )
                
                if batch_predict_button:
                    with st.spinner("🔄 Analyzing employee data..."):
                        # Extract only required features in correct order
                        input_features = df[BEST_FEATURES].copy()
                        
                        # Make predictions
                        predictions = model.predict(input_features)
                        prediction_probabilities = model.predict_proba(input_features)
                        
                        # Create result dataframe with all original columns
                        result_df = df.copy()
                        
                        # Add prediction column with labels
                        result_df[prediction_column_name] = [prediction_labels[p] for p in predictions]
                        
                        # Add probability columns if selected
                        if include_probabilities:
                            result_df[f"{prediction_column_name}_Probability_Stay"] = (prediction_probabilities[:, 0] * 100).round(2)
                            result_df[f"{prediction_column_name}_Probability_Leave"] = (prediction_probabilities[:, 1] * 100).round(2)
                    
                    st.balloons()
                    st.success("🎉 Predictions generated successfully!")
                    
                    st.markdown("---")
                    st.markdown('<h3 style="font-family: Poppins; color: #2c3e50; font-weight: 600;">📊 Analysis Results</h3>', unsafe_allow_html=True)
                    
                    # Summary statistics
                    leaving_count = sum(predictions == 1)
                    staying_count = sum(predictions == 0)
                    leaving_percentage = (leaving_count / len(predictions)) * 100
                    staying_percentage = (staying_count / len(predictions)) * 100
                    
                    col1, col2, col3, col4 = st.columns(4, gap="medium")
                    
                    with col1:
                        st.markdown(f"""
                        <div class="stats-card">
                            <h4>👥 Total Employees</h4>
                            <div class="number">{len(predictions):,}</div>
                            <p style="color: #6c757d; font-size: 0.9rem;">Analyzed</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="stats-card" style="border-top: 4px solid #dc3545;">
                            <h4>🚪 Likely to Leave</h4>
                            <div class="number" style="color: #dc3545;">{leaving_count:,}</div>
                            <p style="color: #6c757d; font-size: 0.9rem;">({leaving_percentage:.1f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="stats-card" style="border-top: 4px solid #28a745;">
                            <h4>🏢 Likely to Stay</h4>
                            <div class="number" style="color: #28a745;">{staying_count:,}</div>
                            <p style="color: #6c757d; font-size: 0.9rem;">({staying_percentage:.1f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        avg_leave_prob = prediction_probabilities[:, 1].mean() * 100
                        risk_color = "#dc3545" if avg_leave_prob > 50 else "#ffc107" if avg_leave_prob > 30 else "#28a745"
                        st.markdown(f"""
                        <div class="stats-card" style="border-top: 4px solid {risk_color};">
                            <h4>📈 Avg Risk Level</h4>
                            <div class="number" style="color: {risk_color};">{avg_leave_prob:.1f}%</div>
                            <p style="color: #6c757d; font-size: 0.9rem;">Leave probability</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Visual representation with enhanced progress bars
                    st.markdown("#### 📈 Turnover Distribution")
                    col1, col2 = st.columns(2, gap="large")
                    
                    with col1:
                        st.markdown(f"**🟢 Employees Staying: {staying_percentage:.1f}%**")
                        st.markdown(f"""
                        <div class="progress-bar-container">
                            <div class="progress-bar-green" style="width: {staying_percentage}%;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"**🔴 Employees Leaving: {leaving_percentage:.1f}%**")
                        st.markdown(f"""
                        <div class="progress-bar-container">
                            <div class="progress-bar-red" style="width: {leaving_percentage}%;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show result preview
                    st.markdown("#### 📄 Results Preview")
                    st.dataframe(result_df.head(20), use_container_width=True)
                    
                    # Download section
                    st.markdown("---")
                    st.markdown('<h3 style="font-family: Poppins; color: #2c3e50; font-weight: 600;">📥 Download Your Results</h3>', unsafe_allow_html=True)
                    
                    # Enhanced download layout
                    if include_high_risk_download:
                        col1, col2, col3 = st.columns(3, gap="medium")
                    else:
                        col1, col2 = st.columns(2, gap="large")
                    
                    with col1:
                        csv_data = convert_df_to_csv(result_df)
                        st.download_button(
                            label="📄 Complete Results (CSV)",
                            data=csv_data,
                            file_name="employee_predictions.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="download_csv"
                        )
                    
                    with col2:
                        excel_data = convert_df_to_excel(result_df)
                        st.download_button(
                            label="📊 Complete Results (Excel)",
                            data=excel_data,
                            file_name="employee_predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key="download_excel"
                        )
                    
                    if include_high_risk_download:
                        with col3:
                            # High-risk employees filter
                            high_risk_df = result_df[prediction_probabilities[:, 1] > 0.5]
                            if len(high_risk_df) > 0:
                                high_risk_csv = convert_df_to_csv(high_risk_df)
                                st.download_button(
                                    label=f"🚨 High Risk Only ({len(high_risk_df)})",
                                    data=high_risk_csv,
                                    file_name="high_risk_employees.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                    key="download_high_risk"
                                )
                            else:
                                st.info("🎉 No high-risk employees found!")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your file is properly formatted and contains the required columns.")
    
    else:
        # Show sample data format
        st.markdown("---")
        with st.expander("📋 View Sample Data Format"):
            st.markdown("#### 💡 Example of properly formatted data:")
            sample_data = pd.DataFrame({
                'employee_id': [1001, 1002, 1003, 1004, 1005],
                'satisfaction_level': [0.38, 0.80, 0.11, 0.72, 0.37],
                'time_spend_company': [3, 5, 4, 3, 2],
                'average_monthly_hours': [157, 262, 272, 223, 159],
                'number_project': [2, 5, 7, 5, 2],
                'last_evaluation': [0.53, 0.86, 0.88, 0.87, 0.52],
                'department': ['sales', 'IT', 'IT', 'sales', 'hr'],
                'salary': ['low', 'medium', 'medium', 'high', 'low']
            })
            st.dataframe(sample_data, use_container_width=True)
            st.info("💡 **Note:** Only the 5 required columns are used for prediction. Additional columns are preserved in output.")

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Stunning header with animation
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <h1 class="main-header">👥 Employee Turnover Prediction</h1>
        <p class="sub-header">🚀 Powered by AI - Predict Employee Retention with Stunning Accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model silently
    model = load_model_from_huggingface()
    
    if model is None:
        st.markdown("""
        <div class="error-box">
            <h4>❌ Model Loading Failed</h4>
            <p>Unable to load the prediction model. Please check the connection and try again.</p>
        </div>
        """, unsafe_allow_html=True)
        st.info(f"📡 Repository: https://huggingface.co/{HF_REPO_ID}")
        return
    
    # ========================================================================
    # SPECTACULAR TABS
    # ========================================================================
    tab1, tab2 = st.tabs(["🎯 Individual Analysis", "📊 Batch Processing"])
    
    with tab1:
        render_individual_prediction_tab(model)
    
    with tab2:
        render_batch_prediction_tab(model)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
