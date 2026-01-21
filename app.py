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
# EXTREMELY DECORATIVE CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* ===== CSS VARIABLES FOR THEME ===== */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        --danger-gradient: linear-gradient(135deg, #ff0844 0%, #ffb199 100%);
        --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        --warning-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --dark-gradient: linear-gradient(135deg, #1a2a6c 0%, #b21f1f 50%, #fdbb2d 100%);
        --neon-pink: #ff2a6d;
        --neon-blue: #05d9e8;
        --neon-green: #d1f7ff;
        --glass-bg: rgba(255, 255, 255, 0.08);
        --glass-border: rgba(255, 255, 255, 0.18);
    }

    /* ===== ANIMATED BACKGROUND ===== */
    body {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        position: relative;
        overflow-x: hidden;
    }
    
    body::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 10% 20%, rgba(255, 42, 109, 0.1) 0%, transparent 20%),
            radial-gradient(circle at 90% 80%, rgba(5, 217, 232, 0.1) 0%, transparent 20%),
            radial-gradient(circle at 30% 60%, rgba(209, 247, 255, 0.05) 0%, transparent 15%);
        animation: floatingParticles 20s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes floatingParticles {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        33% { transform: translate(20px, -20px) rotate(120deg); }
        66% { transform: translate(-10px, 20px) rotate(240deg); }
    }
    
    /* ===== CENTER CONTAINER WITH GLASS EFFECT ===== */
    .block-container {
        max-width: 1400px !important;
        padding: 3rem 4rem !important;
        margin: 0 auto !important;
        background: var(--glass-bg);
        backdrop-filter: blur(12px);
        border-radius: 30px;
        border: 1px solid var(--glass-border);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        animation: containerGlow 4s ease-in-out infinite alternate;
    }
    
    @keyframes containerGlow {
        from { box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37); }
        to { box-shadow: 0 8px 40px 0 rgba(255, 42, 109, 0.4), 0 8px 60px 0 rgba(5, 217, 232, 0.3); }
    }
    
    /* ===== MAIN HEADER - GRADIENT TEXT ANIMATION ===== */
    .main-header {
        font-size: 4rem;
        font-weight: 900;
        background: var(--primary-gradient);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        animation: gradientShift 3s ease infinite;
        text-transform: uppercase;
        letter-spacing: 3px;
        position: relative;
        padding-bottom: 1rem;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 300px;
        height: 5px;
        background: var(--primary-gradient);
        border-radius: 5px;
        animation: borderGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes borderGlow {
        from { box-shadow: 0 0 10px var(--neon-pink); }
        to { box-shadow: 0 0 20px var(--neon-blue), 0 0 30px var(--neon-pink); }
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: var(--neon-green);
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
        text-shadow: 0 0 10px rgba(209, 247, 255, 0.5);
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* ===== SECTION HEADER - NEON EFFECT ===== */
    .section-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: white;
        text-align: center;
        margin: 3rem 0 2rem 0;
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
        display: inline-block;
        width: 100%;
    }
    
    .section-header::before,
    .section-header::after {
        content: '✨';
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        animation: sparkle 2s ease-in-out infinite;
    }
    
    .section-header::before { left: 20%; }
    .section-header::after { right: 20%; }
    
    @keyframes sparkle {
        0%, 100% { opacity: 1; transform: translateY(-50%) scale(1); }
        50% { opacity: 0.5; transform: translateY(-50%) scale(1.5); }
    }
    
    /* ===== FEATURE CARDS - GLASSMORPHISM WITH ANIMATED BORDER ===== */
    .feature-card {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid var(--glass-border);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        margin-bottom: 2rem;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: var(--primary-gradient);
        background-size: 400% 400%;
        border-radius: 20px;
        z-index: -1;
        animation: gradientShift 3s ease infinite;
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    .feature-card:hover::before {
        opacity: 1;
    }
    
    .feature-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 40px rgba(255, 42, 109, 0.3), 0 0 60px rgba(5, 217, 232, 0.2);
    }
    
    .feature-card span {
        font-size: 1.4rem !important;
        font-weight: 700;
        color: white;
        display: block;
        margin-bottom: 1rem;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    
    /* ===== TABS - ENHANCED WITH GRADIENTS ===== */
    .stTabs {
        margin-top: 3rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 25px;
        justify-content: center;
        background: transparent;
        padding: 20px 0;
        border-bottom: none !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 100px !important;
        min-width: 350px !important;
        padding: 0 40px !important;
        background: var(--glass-bg) !important;
        backdrop-filter: blur(5px);
        border-radius: 20px 20px 0 0 !important;
        font-weight: 900 !important;
        font-size: 1.6rem !important;
        color: var(--neon-green) !important;
        border: 2px solid var(--glass-border) !important;
        border-bottom: none !important;
        transition: all 0.4s ease !important;
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stTabs [data-baseweb="tab"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: var(--primary-gradient);
        opacity: 0;
        transition: opacity 0.4s ease;
        z-index: -1;
    }
    
    .stTabs [data-baseweb="tab"]:hover::before {
        opacity: 0.3;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: white !important;
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(255, 42, 109, 0.4);
        border-color: var(--neon-pink) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-gradient) !important;
        color: white !important;
        border-color: var(--neon-pink) !important;
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(255, 42, 109, 0.5), 0 0 40px rgba(5, 217, 232, 0.4);
        animation: tabPulse 2s ease-in-out infinite alternate;
    }
    
    .stTabs [aria-selected="true"]::before {
        opacity: 1;
    }
    
    @keyframes tabPulse {
        from { box-shadow: 0 10px 30px rgba(255, 42, 109, 0.5), 0 0 40px rgba(5, 217, 232, 0.4); }
        to { box-shadow: 0 10px 40px rgba(255, 42, 109, 0.7), 0 0 60px rgba(5, 217, 232, 0.6); }
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 3rem;
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border-radius: 0 0 20px 20px;
        border: 1px solid var(--glass-border);
        border-top: none;
        padding: 3rem 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
    }
    
    /* ===== PREDICTION BOXES - ENHANCED ===== */
    .prediction-box {
        padding: 3rem;
        border-radius: 25px;
        text-align: center;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        border: 2px solid;
        animation: boxEntrance 0.8s ease-out;
        backdrop-filter: blur(10px);
    }
    
    @keyframes boxEntrance {
        from { opacity: 0; transform: scale(0.8) translateY(30px); }
        to { opacity: 1; transform: scale(1) translateY(0); }
    }
    
    .stay-prediction {
        background: rgba(40, 167, 69, 0.15);
        border-color: var(--neon-green);
        box-shadow: 0 0 40px rgba(40, 167, 69, 0.3);
    }
    
    .stay-prediction::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(40, 167, 69, 0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    .leave-prediction {
        background: rgba(220, 53, 69, 0.15);
        border-color: var(--neon-pink);
        box-shadow: 0 0 40px rgba(255, 42, 109, 0.4);
    }
    
    .leave-prediction::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 42, 109, 0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .prediction-box h1 {
        font-size: 3.5rem;
        font-weight: 900;
        margin: 0;
        position: relative;
        z-index: 1;
        text-shadow: 0 0 20px currentColor;
        animation: textGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes textGlow {
        from { text-shadow: 0 0 20px currentColor; }
        to { text-shadow: 0 0 30px currentColor, 0 0 40px currentColor; }
    }
    
    /* ===== METRIC CARDS - GLASSMORPHISM ===== */
    .metric-card {
        background: var(--glass-bg);
        backdrop-filter: blur(5px);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid var(--neon-blue);
        margin: 1rem 0;
        transition: all 0.3s ease;
        border: 1px solid var(--glass-border);
    }
    
    .metric-card:hover {
        transform: translateX(10px);
        box-shadow: 0 10px 25px rgba(5, 217, 232, 0.3);
    }
    
    /* ===== STATS CARDS - ENHANCED ===== */
    .stats-card {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        text-align: center;
        border: 1px solid var(--glass-border);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stats-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: var(--primary-gradient);
    }
    
    .stats-card:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 15px 40px rgba(255, 42, 109, 0.4), 0 0 60px rgba(5, 217, 232, 0.3);
    }
    
    .stats-card h3 {
        color: white;
        margin-bottom: 1rem;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    
    .stats-card .number {
        font-size: 3rem;
        font-weight: 900;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: numberPulse 1.5s ease-in-out infinite alternate;
    }
    
    @keyframes numberPulse {
        from { transform: scale(1); }
        to { transform: scale(1.1); }
    }
    
    /* ===== PROGRESS BARS - ENHANCED ===== */
    .progress-bar-container {
        width: 100%;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        margin: 10px 0 20px 0;
        height: 35px;
        overflow: hidden;
        position: relative;
        border: 1px solid var(--glass-border);
    }
    
    .progress-bar-green, .progress-bar-red {
        height: 100%;
        border-radius: 15px;
        transition: width 1s cubic-bezier(0.22, 0.61, 0.36, 1);
        position: relative;
        overflow: hidden;
        box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.2);
    }
    
    .progress-bar-green {
        background: linear-gradient(90deg, #11998e, #38ef7d, #11998e);
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
    }
    
    .progress-bar-red {
        background: linear-gradient(90deg, #ff0844, #ffb199, #ff0844);
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    .progress-bar-container::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        animation: shimmer 2s infinite;
    }
    
    /* ===== ENHANCED PREDICT BUTTON - ULTRA DECORATIVE ===== */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, 
            #ff0080, #ff8c00, #ffd700, #00ff88, #00c3ff, #ff0080);
        background-size: 600% 600%;
        color: white !important;
        font-size: 1.8rem;
        font-weight: 900 !important;
        padding: 1.8rem 3rem;
        border-radius: 60px;
        border: none !important;
        outline: none !important;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 0 30px rgba(255, 0, 128, 0.6),
            0 0 60px rgba(255, 140, 0, 0.5),
            0 0 90px rgba(0, 195, 255, 0.4);
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: 
            gradientShift 4s ease infinite,
            pulse 3s ease-in-out infinite,
            float 6s ease-in-out infinite;
        text-transform: uppercase;
        letter-spacing: 4px;
        text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.5);
        margin: 2rem 0;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.6), transparent);
        transition: left 0.7s ease;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button::after {
        content: '✨';
        position: absolute;
        font-size: 2rem;
        right: 30px;
        animation: sparkleIcon 2s ease-in-out infinite;
    }
    
    @keyframes sparkleIcon {
        0%, 100% { opacity: 1; transform: scale(1) rotate(0deg); }
        50% { opacity: 0.7; transform: scale(1.5) rotate(180deg); }
    }
    
    .stButton>button:hover {
        transform: translateY(-8px) scale(1.05);
        box-shadow: 
            0 0 50px rgba(255, 0, 128, 0.8),
            0 0 100px rgba(255, 140, 0, 0.7),
            0 0 150px rgba(0, 195, 255, 0.6);
        animation-play-state: paused;
    }
    
    .stButton>button:active {
        transform: translateY(2px) scale(0.98);
        box-shadow: 
            0 0 20px rgba(255, 0, 128, 0.4),
            0 0 40px rgba(255, 140, 0, 0.3);
    }
    
    .stButton>button:focus,
    .stButton>button:focus-visible {
        outline: none !important;
        border: none !important;
        box-shadow: 
            0 0 30px rgba(255, 0, 128, 0.6),
            0 0 60px rgba(255, 140, 0, 0.5),
            0 0 90px rgba(0, 195, 255, 0.4);
    }
    
    /* ===== ENHANCED EXPANDER - BLUE THEME ===== */
    div[data-testid="stExpander"] {
        border: none !important;
        border-radius: 20px !important;
        margin: 2rem 0;
        overflow: hidden;
    }
    
    div[data-testid="stExpander"] details {
        border: none !important;
    }
    
    div[data-testid="stExpander"] details summary {
        background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 50%, #3E7ABF 100%) !important;
        color: white !important;
        border-radius: 15px !important;
        padding: 1.5rem 2rem !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    div[data-testid="stExpander"] details summary::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s ease;
    }
    
    div[data-testid="stExpander"] details summary:hover::before {
        left: 100%;
    }
    
    div[data-testid="stExpander"] details summary:hover {
        background: linear-gradient(135deg, #2E5A8F 0%, #3E7ABF 50%, #4E8ADF 100%) !important;
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(30, 58, 95, 0.4);
    }
    
    div[data-testid="stExpander"] details summary svg {
        color: white !important;
        fill: white !important;
        transition: transform 0.4s ease;
    }
    
    div[data-testid="stExpander"] details[open] summary svg {
        transform: rotate(180deg);
    }
    
    div[data-testid="stExpander"] details[open] summary {
        border-radius: 15px 15px 0 0 !important;
    }
    
    div[data-testid="stExpander"] details > div {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border) !important;
        border-top: none !important;
        border-radius: 0 0 20px 20px !important;
        padding: 2rem !important;
    }
    
    /* ===== UPLOAD SECTION - ENHANCED ===== */
    .upload-section {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border-radius: 25px;
        padding: 3rem;
        border: 3px dashed var(--neon-blue);
        margin: 2rem 0;
        text-align: center;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .upload-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: var(--primary-gradient);
        opacity: 0;
        transition: opacity 0.4s ease;
        z-index: -1;
    }
    
    .upload-section:hover::before {
        opacity: 0.1;
    }
    
    .upload-section:hover {
        border-color: var(--neon-pink);
        transform: scale(1.02);
        box-shadow: 0 15px 40px rgba(5, 217, 232, 0.3);
    }
    
    /* ===== SETTINGS CARD - ENHANCED ===== */
    .settings-card {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid var(--glass-border);
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .settings-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(5, 217, 232, 0.3);
    }
    
    .settings-card h4 {
        color: white;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    
    /* ===== INFO/ERROR/SUCCESS BOXES - ENHANCED ===== */
    .info-box, .error-box, .success-box {
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
        border: 1px solid;
        backdrop-filter: blur(5px);
    }
    
    .info-box {
        background: rgba(255, 193, 7, 0.1);
        border-color: #ffc107;
        color: #ffc107;
    }
    
    .error-box {
        background: rgba(220, 53, 69, 0.1);
        border-color: #dc3545;
        color: #dc3545;
    }
    
    .success-box {
        background: rgba(40, 167, 69, 0.1);
        border-color: #28a745;
        color: #28a745;
    }
    
    .info-box::before, .error-box::before, .success-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: currentColor;
    }
    
    .info-box::after, .error-box::after, .success-box::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, currentColor, transparent);
        opacity: 0.1;
        animation: shimmer 3s infinite;
    }
    
    /* ===== INPUT FIELDS - CUSTOM STYLING ===== */
    .stNumberInput input, .stSelectbox select, .stTextInput input {
        background: var(--glass-bg) !important;
        border: 2px solid var(--glass-border) !important;
        border-radius: 15px !important;
        color: white !important;
        font-weight: 500;
        padding: 0.8rem 1rem !important;
        transition: all 0.3s ease;
    }
    
    .stNumberInput input:focus, .stSelectbox select:focus, .stTextInput input:focus {
        border-color: var(--neon-pink) !important;
        box-shadow: 0 0 20px rgba(255, 42, 109, 0.4) !important;
        transform: scale(1.05);
    }
    
    .stSlider > div > div > div > div {
        background: var(--primary-gradient) !important;
        border-radius: 10px !important;
    }
    
    /* ===== CUSTOM SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 10px;
        border: 2px solid rgba(0, 0, 0, 0.3);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--neon-pink);
    }
    
    /* ===== DOWNLOAD BUTTONS - ENHANCED ===== */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #28a745, #20c997, #28a745) !important;
        background-size: 200% 200% !important;
        animation: gradientShift 2s ease infinite !important;
        border-radius: 20px !important;
        font-weight: 700 !important;
        padding: 1rem 1.5rem !important;
        transition: all 0.3s ease !important;
        border: none !important;
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-5px) scale(1.05) !important;
        box-shadow: 0 10px 25px rgba(40, 167, 69, 0.5) !important;
    }
    
    /* ===== DATAFRAME - ENHANCED ===== */
    .dataframe {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(10px);
        border-radius: 15px !important;
        border: 1px solid var(--glass-border) !important;
        overflow: hidden;
    }
    
    .dataframe th {
        background: var(--primary-gradient) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 1rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .dataframe td {
        padding: 1rem !important;
        color: white !important;
        border-bottom: 1px solid var(--glass-border) !important;
    }
    
    .dataframe tr:hover {
        background: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* ===== CHECKBOX ===== */
    .stCheckbox {
        padding: 1rem 0;
    }
    
    .stCheckbox label {
        font-size: 1.1rem;
        font-weight: 500;
        color: white;
    }
    
    .stCheckbox>div>div>div {
        background: var(--neon-pink) !important;
        border-color: var(--neon-pink) !important;
    }
    
    /* ===== SELECTBOX DROPDOWN ===== */
    .stSelectbox div[role="listbox"] {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border) !important;
        border-radius: 15px !important;
    }
    
    .stSelectbox div[role="option"] {
        color: white !important;
        font-weight: 500;
    }
    
    .stSelectbox div[role="option"]:hover {
        background: var(--primary-gradient) !important;
    }
    
    /* ===== FILE UPLOADER ===== */
    .stFileUploader {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(10px);
        border: 2px dashed var(--neon-blue) !important;
        border-radius: 25px !important;
        padding: 2rem !important;
        transition: all 0.4s ease;
    }
    
    .stFileUploader:hover {
        border-color: var(--neon-pink) !important;
        transform: scale(1.02);
    }
    
    /* ===== REMOVE STREAMLIT BRANDING ===== */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .block-container {
            padding: 1.5rem !important;
        }
        .main-header {
            font-size: 2.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            min-width: 100% !important;
            font-size: 1.2rem !important;
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
    st.markdown('<h2 class="section-header">📝 Enter Employee Information</h2>', unsafe_allow_html=True)
    
    # Initialize session state for syncing slider and number input
    if 'satisfaction_level' not in st.session_state:
        st.session_state.satisfaction_level = 0.5
    if 'last_evaluation' not in st.session_state:
        st.session_state.last_evaluation = 0.7
    
    # ========================================================================
    # ROW 1: Satisfaction Level & Last Evaluation (side by side with feature cards)
    # ========================================================================
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <span>😊 <strong>Satisfaction Level</strong></span>
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
                <span>📊 <strong>Last Evaluation</strong></span>
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
        
        # ========================================================================
        # ROW 2: Years at Company, Number of Projects, Average Monthly Hours (3 columns)
        # ========================================================================
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <span>📅 <strong>Years at Company</strong></span>
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
                <span>📁 <strong>Number of Projects</strong></span>
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
                <span>⏰ <strong>Avg. Monthly Hours</strong></span>
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
    st.markdown("---")
    
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
        st.subheader("🎯 Prediction Results")
        
        # Results in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0:
                st.markdown("""
                <div class="prediction-box stay-prediction">
                    <h1>✅ STAY</h1>
                    <p style="font-size: 1.5rem; margin-top: 1rem; font-weight: 600;">
                        Employee is likely to <strong>STAY</strong> with the company
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-box leave-prediction">
                    <h1>⚠️ LEAVE</h1>
                    <p style="font-size: 1.5rem; margin-top: 1rem; font-weight: 600;">
                        Employee is likely to <strong>LEAVE</strong> the company
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Prediction Probabilities")
            
            # Stay probability with GREEN bar
            st.markdown(f"<p style='font-size: 1.2rem; font-weight: 600;'>Probability of Staying: <span style='color: var(--neon-green);'>{prob_stay:.1f}%</span></p>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="progress-bar-container">
                <div class="progress-bar-green" style="width: {prob_stay}%;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Leave probability with RED bar
            st.markdown(f"<p style='font-size: 1.2rem; font-weight: 600;'>Probability of Leaving: <span style='color: var(--neon-pink);'>{prob_leave:.1f}%</span></p>", unsafe_allow_html=True)
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
    
    # ========================================================================
    # REQUIRED COLUMNS INFO - NOW AS DROPDOWN/EXPANDER
    # ========================================================================
    with st.expander("📋 Required Columns in Your File (Click to Expand)"):
        st.markdown("""
        <div class="info-box">
            <p>Your uploaded file <strong>must contain</strong> these columns with <strong>exact names</strong>:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display required features in a nice format
        col1, col2 = st.columns(2)
        with col1:
            for feature in BEST_FEATURES[:3]:
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{feature}</strong><br/>
                    <small style="color: #ccc;">{FEATURE_DESCRIPTIONS[feature]}</small>
                </div>
                """, unsafe_allow_html=True)
        with col2:
            for feature in BEST_FEATURES[3:]:
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{feature}</strong><br/>
                    <small style="color: #ccc;">{FEATURE_DESCRIPTIONS[feature]}</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.info("💡 **Tip:** Your file can contain additional columns (like employee_id, department, etc.). They will be preserved in the output but won't be used for prediction.")
    
    st.markdown("---")
    
    # ========================================================================
    # FILE UPLOAD SECTION
    # ========================================================================
    st.markdown("### 📁 Upload Your Data")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="settings-card">
            <h4>⚙️ File Settings</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # File format selection
        file_format = st.selectbox(
            "Select file format",
            options=["CSV", "Excel (.xlsx)"],
            index=0,
            help="Choose the format of your data file",
            key="file_format"
        )
    
    with col2:
        # File uploader based on format
        if file_format == "CSV":
            uploaded_file = st.file_uploader(
                "Upload your CSV file",
                type=["csv"],
                help="Upload a CSV file containing employee data",
                key="csv_uploader"
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload your Excel file",
                type=["xlsx", "xls"],
                help="Upload an Excel file containing employee data",
                key="excel_uploader"
            )
    
    st.markdown("---")
    
    # ========================================================================
    # PREDICTION OUTPUT SETTINGS
    # ========================================================================
    st.markdown("### ⚙️ Prediction Output Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="settings-card">
            <h4>📝 Column Name for Predictions</h4>
        </div>
        """, unsafe_allow_html=True)
        
        column_name_option = st.selectbox(
            "Select prediction column name",
            options=["Prediction", "Churn", "Will_Leave", "Turnover", "Custom"],
            index=0,
            help="Choose the column name where predictions will be stored",
            key="column_name_option"
        )
        
        if column_name_option == "Custom":
            custom_column_name = st.text_input(
                "Enter custom column name",
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
            <h4>🏷️ Prediction Labels</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Label options - Model predicts 1 for leaving, 0 for staying
        label_option = st.selectbox(
            "Select prediction labels",
            options=["Leave / Stay", "Yes / No", "Churn / Not Churn", "1 / 0", "True / False", "Custom"],
            index=0,
            help="Choose how predictions should be labeled (Leaving / Staying format)",
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
                    "Label for LEAVING",
                    value="Leaving",
                    help="Label when employee is predicted to leave",
                    key="custom_leave_label"
                )
            with custom_col2:
                custom_stay_label = st.text_input(
                    "Label for STAYING",
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
        
        st.info(f"📌 **Label Preview:** Leaving → '{prediction_labels[1]}' | Staying → '{prediction_labels[0]}'")
    
    # ========================================================================
    # PROBABILITY COLUMNS OPTION
    # ========================================================================
    st.markdown("---")
    st.markdown("### 📊 Additional Output Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="settings-card">
            <h4>🎯 Probability Columns</h4>
        </div>
        """, unsafe_allow_html=True)
        
        include_probabilities = st.checkbox(
            "Include prediction probabilities in output",
            value=True,
            help="Add columns showing the probability (%) of staying and leaving for each employee",
            key="include_probabilities"
        )
        
        if include_probabilities:
            st.success("✅ Two additional columns will be added: `Probability_Stay` and `Probability_Leave`")
        else:
            st.info("ℹ️ Only the prediction label column will be added to the output")
    
    with col2:
        st.markdown("""
        <div class="settings-card">
            <h4>⚠️ High Risk Filter</h4>
        </div>
        """, unsafe_allow_html=True)
        
        include_high_risk_download = st.checkbox(
            "Enable high-risk employees download",
            value=True,
            help="Provide a separate download option for employees with >50% probability of leaving",
            key="include_high_risk"
        )
        
        if include_high_risk_download:
            st.success("✅ A separate download button for high-risk employees will be available")
        else:
            st.info("ℹ️ Only full results download will be available")
    
    # ========================================================================
    # PROCESS UPLOADED FILE
    # ========================================================================
    if uploaded_file is not None:
        st.markdown("---")
        st.markdown("### 📄 Uploaded Data Preview")
        
        try:
            # Read the file based on format
            if file_format == "CSV":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Display file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="stats-card">
                    <h4>Total Rows</h4>
                    <div class="number">{len(df):,}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="stats-card">
                    <h4>Total Columns</h4>
                    <div class="number">{len(df.columns)}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                # Check for required columns
                available_features = [col for col in BEST_FEATURES if col in df.columns]
                st.markdown(f"""
                <div class="stats-card">
                    <h4>Required Cols Found</h4>
                    <div class="number">{len(available_features)}/{len(BEST_FEATURES)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Show data preview
            st.dataframe(df.head(10), use_container_width=True)
            
            # Check for missing required columns
            missing_columns = [col for col in BEST_FEATURES if col not in df.columns]
            
            if missing_columns:
                st.markdown(f"""
                <div class="error-box">
                    <h4>❌ Missing Required Columns</h4>
                    <p>The following required columns are missing from your file:</p>
                    <ul>
                        {''.join([f'<li><strong>{col}</strong>: {FEATURE_DESCRIPTIONS[col]}</li>' for col in missing_columns])}
                    </ul>
                    <p>Please ensure your file contains all required columns with exact names.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                    <h4>✅ All Required Columns Found!</h4>
                    <p>Your file contains all necessary columns for prediction. Click the button below to generate predictions.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show which columns will be used
                with st.expander("🔍 View columns being used for prediction"):
                    for feature in BEST_FEATURES:
                        sample_values = df[feature].head(3).tolist()
                        st.write(f"• **{feature}**: Sample values → {sample_values}")
                
                st.markdown("---")
                
                # Prediction button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    batch_predict_button = st.button(
                        "🔮 Generate Batch Predictions",
                        use_container_width=True,
                        key="batch_predict"
                    )
                
                if batch_predict_button:
                    with st.spinner("🔄 Processing predictions... ⏳"):
                        # Extract only required features in correct order
                        input_features = df[BEST_FEATURES].copy()
                        
                        # Make predictions
                        predictions = model.predict(input_features)
                        prediction_probabilities = model.predict_proba(input_features)
                        
                        # Create result dataframe with all original columns
                        result_df = df.copy()
                        
                        # Add prediction column with labels
                        result_df[prediction_column_name] = [prediction_labels[p] for p in predictions]
                        
                        # Add probability columns only if user selected this option
                        if include_probabilities:
                            result_df[f"{prediction_column_name}_Probability_Stay"] = (prediction_probabilities[:, 0] * 100).round(2)
                            result_df[f"{prediction_column_name}_Probability_Leave"] = (prediction_probabilities[:, 1] * 100).round(2)
                    
                    st.success("✅ Predictions generated successfully! 🎉")
                    
                    st.markdown("---")
                    st.markdown("### 📊 Prediction Results")
                    
                    # Summary statistics
                    leaving_count = sum(predictions == 1)
                    staying_count = sum(predictions == 0)
                    leaving_percentage = (leaving_count / len(predictions)) * 100
                    staying_percentage = (staying_count / len(predictions)) * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="stats-card">
                            <h4>Total Employees</h4>
                            <div class="number">{len(predictions):,}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="stats-card" style="border-top-color: #dc3545;">
                            <h4>Predicted to Leave</h4>
                            <div class="number" style="color: #dc3545;">{leaving_count:,}</div>
                            <p style="color: #666;">({leaving_percentage:.1f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="stats-card" style="border-top-color: #28a745;">
                            <h4>Predicted to Stay</h4>
                            <div class="number" style="color: #28a745;">{staying_count:,}</div>
                            <p style="color: #666;">({staying_percentage:.1f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        avg_leave_prob = prediction_probabilities[:, 1].mean() * 100
                        st.markdown(f"""
                        <div class="stats-card" style="border-top-color: #ffc107;">
                            <h4>Avg. Leave Probability</h4>
                            <div class="number" style="color: #ffc107;">{avg_leave_prob:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Visual representation
                    st.markdown("#### 📈 Turnover Distribution")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Staying:** {staying_percentage:.1f}%")
                        st.markdown(f"""
                        <div class="progress-bar-container">
                            <div class="progress-bar-green" style="width: {staying_percentage}%;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.write(f"**Leaving:** {leaving_percentage:.1f}%")
                        st.markdown(f"""
                        <div class="progress-bar-container">
                            <div class="progress-bar-red" style="width: {leaving_percentage}%;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show result preview
                    st.markdown("#### 📄 Result Data Preview")
                    st.dataframe(result_df.head(20), use_container_width=True)
                    
                    # Download section
                    st.markdown("---")
                    st.markdown("### 📥 Download Results")
                    
                    # Determine number of columns based on options
                    if include_high_risk_download:
                        col1, col2, col3 = st.columns([1, 1, 1])
                    else:
                        col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        csv_data = convert_df_to_csv(result_df)
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv_data,
                            file_name="employee_predictions.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="download_csv"
                        )
                    
                    with col2:
                        excel_data = convert_df_to_excel(result_df)
                        st.download_button(
                            label="📥 Download as Excel",
                            data=excel_data,
                            file_name="employee_predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key="download_excel"
                        )
                    
                    if include_high_risk_download:
                        with col3:
                            # Download only high-risk employees (probability > 50%)
                            high_risk_df = result_df[prediction_probabilities[:, 1] > 0.5]
                            if len(high_risk_df) > 0:
                                high_risk_csv = convert_df_to_csv(high_risk_df)
                                st.download_button(
                                    label=f"📥 High Risk Only ({len(high_risk_df)})",
                                    data=high_risk_csv,
                                    file_name="high_risk_employees.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                    key="download_high_risk"
                                )
                            else:
                                st.info("No high-risk employees found")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please ensure your file is properly formatted and not corrupted.")
    
    else:
        # Show sample data format
        st.markdown("---")
        with st.expander("📋 View Sample Data Format"):
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
            st.info("💡 Note: Only the 5 required columns will be used for prediction. Other columns will be preserved in the output.")

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">👥 Employee Turnover Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict whether employees are likely to leave the company</p>', unsafe_allow_html=True)
    
    # Load model silently
    model = load_model_from_huggingface()
    
    if model is None:
        st.error("❌ Failed to load model. Please check the Hugging Face repository.")
        st.info(f"Repository: https://huggingface.co/{HF_REPO_ID}")
        return
    
    # ========================================================================
    # CREATE TABS
    # ========================================================================
    tab1, tab2 = st.tabs(["📝 Individual Prediction", "📊 Batch Prediction"])
    
    with tab1:
        render_individual_prediction_tab(model)
    
    with tab2:
        render_batch_prediction_tab(model)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
