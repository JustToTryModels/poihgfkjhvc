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
# ULTRA STUNNING CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* ========== IMPORT GOOGLE FONTS ========== */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&family=Poppins:wght@300;400;500;600;700;800;900&display=swap');
    
    /* ========== ANIMATED BACKGROUND ========== */
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #0f0c29);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Animated particles overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(255,0,128,0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(0,255,255,0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(255,255,0,0.05) 0%, transparent 50%),
            radial-gradient(circle at 60% 60%, rgba(128,0,255,0.1) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    
    /* ========== MAIN CONTAINER STYLING ========== */
    .block-container {
        max-width: 1400px !important;
        padding: 2rem 3rem !important;
        margin: 0 auto !important;
    }
    
    /* ========== STUNNING GLASSMORPHISM MAIN CARD ========== */
    .main .block-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 
            0 25px 50px -12px rgba(0, 0, 0, 0.5),
            inset 0 1px 1px rgba(255, 255, 255, 0.1);
    }
    
    /* ========== MEGA HEADER WITH ANIMATED GRADIENT TEXT ========== */
    .main-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(
            135deg, 
            #f093fb 0%, 
            #f5576c 25%, 
            #4facfe 50%, 
            #00f2fe 75%,
            #f093fb 100%
        );
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmerText 4s ease-in-out infinite;
        text-shadow: 0 0 80px rgba(240, 147, 251, 0.5);
        letter-spacing: 3px;
        position: relative;
    }
    
    .main-header::before {
        content: '👥';
        font-size: 3rem;
        margin-right: 15px;
        animation: bounce 2s ease-in-out infinite;
        display: inline-block;
    }
    
    @keyframes shimmerText {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        25% { transform: translateY(-10px) rotate(-5deg); }
        50% { transform: translateY(0) rotate(0deg); }
        75% { transform: translateY(-5px) rotate(5deg); }
    }
    
    /* ========== SUB HEADER ========== */
    .sub-header {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.5rem;
        font-weight: 500;
        text-align: center;
        margin-bottom: 3rem;
        color: rgba(255, 255, 255, 0.7);
        letter-spacing: 5px;
        text-transform: uppercase;
        animation: fadeInUp 1s ease-out 0.3s both;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* ========== SECTION HEADERS ========== */
    .section-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin: 2rem 0;
        color: #fff;
        text-shadow: 
            0 0 10px rgba(79, 172, 254, 0.5),
            0 0 20px rgba(79, 172, 254, 0.3),
            0 0 30px rgba(79, 172, 254, 0.2);
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 150px;
        height: 4px;
        background: linear-gradient(90deg, transparent, #4facfe, #00f2fe, #4facfe, transparent);
        border-radius: 2px;
    }
    
    /* ========== MEGA TAB STYLING ========== */
    .stTabs {
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 30px;
        justify-content: center;
        background: transparent;
        padding: 20px 0;
        border-bottom: none;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 100px !important;
        min-width: 380px !important;
        padding: 0 60px !important;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        color: rgba(255, 255, 255, 0.8) !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        position: relative;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.2),
            transparent
        );
        transition: left 0.5s;
    }
    
    .stTabs [data-baseweb="tab"]:hover::before {
        left: 100%;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.3) 0%, rgba(0, 242, 254, 0.2) 100%) !important;
        transform: translateY(-8px) scale(1.02);
        box-shadow: 
            0 20px 40px rgba(79, 172, 254, 0.3),
            0 0 60px rgba(0, 242, 254, 0.2),
            inset 0 1px 1px rgba(255, 255, 255, 0.2);
        border-color: rgba(79, 172, 254, 0.5) !important;
        color: #fff !important;
    }
    
    .stTabs [data-baseweb="tab"] p {
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        color: inherit !important;
        margin: 0 !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
        background-size: 200% 200%;
        animation: activeTabGlow 3s ease infinite;
        color: white !important;
        border: 2px solid transparent !important;
        box-shadow: 
            0 15px 35px rgba(102, 126, 234, 0.5),
            0 0 60px rgba(118, 75, 162, 0.4),
            0 0 100px rgba(240, 147, 251, 0.2),
            inset 0 1px 1px rgba(255, 255, 255, 0.3);
        transform: translateY(-5px);
    }
    
    @keyframes activeTabGlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stTabs [aria-selected="true"] p {
        color: white !important;
        font-weight: 800 !important;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 30px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab-border"],
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    
    /* ========== STUNNING FEATURE CARDS ========== */
    .feature-card {
        background: linear-gradient(
            135deg, 
            rgba(102, 126, 234, 0.2) 0%, 
            rgba(118, 75, 162, 0.15) 50%,
            rgba(240, 147, 251, 0.1) 100%
        );
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 1.5rem 2rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 
            0 10px 40px rgba(0, 0, 0, 0.2),
            inset 0 1px 1px rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 300% 100%;
        animation: borderGlow 3s linear infinite;
    }
    
    @keyframes borderGlow {
        0% { background-position: 0% 50%; }
        100% { background-position: 300% 50%; }
    }
    
    .feature-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 
            0 25px 60px rgba(102, 126, 234, 0.3),
            0 0 40px rgba(240, 147, 251, 0.2),
            inset 0 1px 1px rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.25);
    }
    
    .feature-card span {
        font-family: 'Poppins', sans-serif;
        font-size: 1.2rem;
        color: #fff;
        font-weight: 600;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    /* ========== PREDICTION BOXES ========== */
    .prediction-box {
        padding: 3rem;
        border-radius: 25px;
        text-align: center;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .stay-prediction {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.3) 0%, rgba(5, 150, 105, 0.2) 100%);
        border: 2px solid rgba(16, 185, 129, 0.5);
        box-shadow: 
            0 20px 60px rgba(16, 185, 129, 0.3),
            0 0 100px rgba(5, 150, 105, 0.2),
            inset 0 1px 1px rgba(255, 255, 255, 0.1);
        animation: pulseGreen 2s ease-in-out infinite;
    }
    
    @keyframes pulseGreen {
        0%, 100% { box-shadow: 0 20px 60px rgba(16, 185, 129, 0.3), 0 0 100px rgba(5, 150, 105, 0.2); }
        50% { box-shadow: 0 25px 80px rgba(16, 185, 129, 0.5), 0 0 150px rgba(5, 150, 105, 0.4); }
    }
    
    .stay-prediction h1 {
        font-family: 'Orbitron', sans-serif;
        font-size: 4rem;
        color: #10b981;
        text-shadow: 0 0 30px rgba(16, 185, 129, 0.8);
        animation: glowTextGreen 2s ease-in-out infinite;
    }
    
    @keyframes glowTextGreen {
        0%, 100% { text-shadow: 0 0 30px rgba(16, 185, 129, 0.8); }
        50% { text-shadow: 0 0 60px rgba(16, 185, 129, 1), 0 0 100px rgba(16, 185, 129, 0.5); }
    }
    
    .stay-prediction p {
        font-family: 'Poppins', sans-serif;
        font-size: 1.4rem;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .leave-prediction {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.3) 0%, rgba(220, 38, 38, 0.2) 100%);
        border: 2px solid rgba(239, 68, 68, 0.5);
        box-shadow: 
            0 20px 60px rgba(239, 68, 68, 0.3),
            0 0 100px rgba(220, 38, 38, 0.2),
            inset 0 1px 1px rgba(255, 255, 255, 0.1);
        animation: pulseRed 2s ease-in-out infinite;
    }
    
    @keyframes pulseRed {
        0%, 100% { box-shadow: 0 20px 60px rgba(239, 68, 68, 0.3), 0 0 100px rgba(220, 38, 38, 0.2); }
        50% { box-shadow: 0 25px 80px rgba(239, 68, 68, 0.5), 0 0 150px rgba(220, 38, 38, 0.4); }
    }
    
    .leave-prediction h1 {
        font-family: 'Orbitron', sans-serif;
        font-size: 4rem;
        color: #ef4444;
        text-shadow: 0 0 30px rgba(239, 68, 68, 0.8);
        animation: glowTextRed 2s ease-in-out infinite;
    }
    
    @keyframes glowTextRed {
        0%, 100% { text-shadow: 0 0 30px rgba(239, 68, 68, 0.8); }
        50% { text-shadow: 0 0 60px rgba(239, 68, 68, 1), 0 0 100px rgba(239, 68, 68, 0.5); }
    }
    
    .leave-prediction p {
        font-family: 'Poppins', sans-serif;
        font-size: 1.4rem;
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* ========== STUNNING PROGRESS BARS ========== */
    .progress-bar-container {
        width: 100%;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        margin: 8px 0 20px 0;
        height: 30px;
        overflow: hidden;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .progress-bar-green {
        height: 100%;
        background: linear-gradient(90deg, #10b981, #34d399, #6ee7b7, #34d399, #10b981);
        background-size: 200% 100%;
        border-radius: 15px;
        animation: shimmerGreen 2s linear infinite;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.5);
    }
    
    @keyframes shimmerGreen {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    .progress-bar-red {
        height: 100%;
        background: linear-gradient(90deg, #ef4444, #f87171, #fca5a5, #f87171, #ef4444);
        background-size: 200% 100%;
        border-radius: 15px;
        animation: shimmerRed 2s linear infinite;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
    }
    
    @keyframes shimmerRed {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    /* ========== STATS CARDS ========== */
    .stats-card {
        background: linear-gradient(
            135deg, 
            rgba(255, 255, 255, 0.1) 0%, 
            rgba(255, 255, 255, 0.05) 100%
        );
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 
            0 15px 35px rgba(0, 0, 0, 0.2),
            inset 0 1px 1px rgba(255, 255, 255, 0.1);
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
        transition: all 0.4s ease;
    }
    
    .stats-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    }
    
    .stats-card:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 25px 50px rgba(102, 126, 234, 0.3),
            inset 0 1px 1px rgba(255, 255, 255, 0.2);
    }
    
    .stats-card h4 {
        font-family: 'Poppins', sans-serif;
        color: rgba(255, 255, 255, 0.8);
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stats-card .number {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stats-card p {
        color: rgba(255, 255, 255, 0.6);
        font-family: 'Poppins', sans-serif;
    }
    
    /* ========== MEGA PREDICT BUTTON ========== */
    .stButton>button {
        width: 100%;
        background: linear-gradient(
            45deg, 
            #ff0080, #ff8c00, #40e0d0, #ff0080, #ff8c00
        );
        background-size: 400% 400%;
        color: white !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 1.5rem;
        font-weight: 800 !important;
        padding: 1.5rem 3rem;
        border-radius: 60px;
        border: none !important;
        outline: none !important;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 10px 40px rgba(255, 0, 128, 0.4),
            0 15px 60px rgba(255, 140, 0, 0.3),
            0 0 80px rgba(64, 224, 208, 0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: gradientShift 3s ease infinite, megaPulse 2s ease-in-out infinite;
        text-transform: uppercase;
        letter-spacing: 4px;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes megaPulse {
        0%, 100% { 
            box-shadow: 
                0 10px 40px rgba(255, 0, 128, 0.4),
                0 15px 60px rgba(255, 140, 0, 0.3),
                0 0 80px rgba(64, 224, 208, 0.2);
            transform: scale(1);
        }
        50% { 
            box-shadow: 
                0 15px 60px rgba(255, 0, 128, 0.6),
                0 20px 80px rgba(255, 140, 0, 0.5),
                0 0 120px rgba(64, 224, 208, 0.4);
            transform: scale(1.02);
        }
    }
    
    .stButton>button:hover {
        background: linear-gradient(
            45deg, 
            #00f5ff, #ff00ff, #ffff00, #00f5ff, #ff00ff
        );
        background-size: 400% 400%;
        transform: translateY(-8px) scale(1.02) !important;
        box-shadow: 
            0 20px 60px rgba(0, 245, 255, 0.5),
            0 30px 100px rgba(255, 0, 255, 0.4),
            0 0 150px rgba(255, 255, 0, 0.3) !important;
        animation: gradientShift 1.5s ease infinite !important;
    }
    
    .stButton>button:active {
        transform: translateY(2px) scale(0.98) !important;
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
            rgba(255, 255, 255, 0.4),
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
    
    @keyframes sparkle {
        0%, 100% { opacity: 1; transform: scale(1) rotate(0deg); }
        50% { opacity: 0.5; transform: scale(1.3) rotate(180deg); }
    }
    
    .stButton>button,
    .stButton>button:hover,
    .stButton>button:active,
    .stButton>button:focus,
    .stButton>button:focus-visible {
        color: white !important;
        outline: none !important;
        border: none !important;
    }
    
    /* ========== DOWNLOAD BUTTONS ========== */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #10b981, #059669, #34d399) !important;
        animation: none !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.4) !important;
    }
    
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #34d399, #10b981, #059669) !important;
        transform: translateY(-5px) scale(1.02) !important;
        box-shadow: 0 15px 50px rgba(16, 185, 129, 0.6) !important;
    }
    
    /* ========== SETTINGS CARD ========== */
    .settings-card {
        background: linear-gradient(
            135deg, 
            rgba(6, 182, 212, 0.2) 0%, 
            rgba(59, 130, 246, 0.15) 100%
        );
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(6, 182, 212, 0.3);
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(6, 182, 212, 0.2);
    }
    
    .settings-card h4 {
        font-family: 'Poppins', sans-serif;
        color: #67e8f9;
        font-weight: 600;
        margin: 0;
        font-size: 1.1rem;
    }
    
    /* ========== INFO/SUCCESS/ERROR BOXES ========== */
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.15) 100%);
        border: 1px solid rgba(16, 185, 129, 0.5);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
    }
    
    .success-box h4 {
        font-family: 'Poppins', sans-serif;
        color: #34d399;
        margin-bottom: 0.5rem;
    }
    
    .success-box p {
        color: rgba(255, 255, 255, 0.8);
        font-family: 'Poppins', sans-serif;
    }
    
    .error-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.15) 100%);
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.2);
    }
    
    .error-box h4 {
        font-family: 'Poppins', sans-serif;
        color: #f87171;
    }
    
    .error-box p, .error-box li {
        color: rgba(255, 255, 255, 0.8);
        font-family: 'Poppins', sans-serif;
    }
    
    /* ========== EXPANDER STYLING ========== */
    div[data-testid="stExpander"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.15) 100%);
        border-radius: 15px !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(139, 92, 246, 0.2);
    }
    
    div[data-testid="stExpander"] details {
        border: none !important;
    }
    
    div[data-testid="stExpander"] details summary {
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7) !important;
        color: white !important;
        border-radius: 15px !important;
        padding: 1rem 1.5rem !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stExpander"] details summary:hover {
        background: linear-gradient(135deg, #8b5cf6, #a855f7, #c084fc) !important;
        box-shadow: 0 5px 20px rgba(139, 92, 246, 0.4);
    }
    
    div[data-testid="stExpander"] details summary svg {
        color: white !important;
        fill: white !important;
    }
    
    div[data-testid="stExpander"] details[open] summary {
        border-radius: 15px 15px 0 0 !important;
    }
    
    div[data-testid="stExpander"] details > div {
        background: rgba(0, 0, 0, 0.2);
        border: none !important;
        border-radius: 0 0 15px 15px !important;
        padding: 1rem;
    }
    
    /* ========== INPUT FIELDS STYLING ========== */
    .stNumberInput, .stSelectbox, .stSlider {
        background: transparent;
    }
    
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        color: white !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.3) !important;
    }
    
    /* ========== SLIDER STYLING ========== */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb) !important;
    }
    
    .stSlider > div > div > div > div > div {
        background: white !important;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
    }
    
    /* ========== CHECKBOX STYLING ========== */
    .stCheckbox label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500 !important;
    }
    
    .stCheckbox > div > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    /* ========== FILE UPLOADER ========== */
    .stFileUploader {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        border: 2px dashed rgba(139, 92, 246, 0.5) !important;
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: rgba(139, 92, 246, 0.8) !important;
        box-shadow: 0 0 30px rgba(139, 92, 246, 0.3);
    }
    
    /* ========== DATAFRAME STYLING ========== */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    
    /* ========== DIVIDER ========== */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.5), rgba(240, 147, 251, 0.5), rgba(79, 172, 254, 0.5), transparent);
        margin: 2rem 0;
    }
    
    /* ========== INFO ALERTS ========== */
    .stAlert {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.15) 100%) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 12px !important;
    }
    
    /* ========== SUBHEADER ========== */
    h2, h3 {
        font-family: 'Poppins', sans-serif !important;
        color: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* ========== LABELS & TEXT ========== */
    label, p, span {
        font-family: 'Poppins', sans-serif !important;
        color: rgba(255, 255, 255, 0.85) !important;
    }
    
    /* ========== FLOATING PARTICLES ANIMATION ========== */
    @keyframes float {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(5deg); }
    }
    
    /* ========== METRIC CARD ========== */
    .metric-card {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.2) 0%, rgba(0, 242, 254, 0.15) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(79, 172, 254, 0.3);
        margin: 0.5rem 0;
        box-shadow: 0 10px 30px rgba(79, 172, 254, 0.2);
    }
    
    /* ========== GLOW TEXT EFFECT ========== */
    .glow-text {
        text-shadow: 
            0 0 10px currentColor,
            0 0 20px currentColor,
            0 0 30px currentColor;
    }
    
    /* ========== RAINBOW BORDER ANIMATION ========== */
    @keyframes rainbowBorder {
        0% { border-color: #ff0080; }
        20% { border-color: #ff8c00; }
        40% { border-color: #40e0d0; }
        60% { border-color: #8b5cf6; }
        80% { border-color: #f093fb; }
        100% { border-color: #ff0080; }
    }
    
    /* ========== HIDE STREAMLIT BRANDING ========== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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

# ============================================================================
# INDIVIDUAL PREDICTION TAB
# ============================================================================
def render_individual_prediction_tab(model):
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
                <span style="font-size: 1.3rem;">😊 <strong>Satisfaction Level</strong></span>
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
                <span style="font-size: 1.3rem;">📊 <strong>Last Evaluation</strong></span>
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
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <span style="font-size: 1.3rem;">📅 <strong>Years at Company</strong></span>
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
                <span style="font-size: 1.3rem;">📁 <strong>Number of Projects</strong></span>
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
                <span style="font-size: 1.3rem;">⏰ <strong>Avg. Monthly Hours</strong></span>
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
        predict_button = st.button("🔮 PREDICT EMPLOYEE TURNOVER", use_container_width=True, key="individual_predict")
    
    if predict_button:
        input_df = pd.DataFrame([input_data])[BEST_FEATURES]
        
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        prob_stay = prediction_proba[0] * 100
        prob_leave = prediction_proba[1] * 100
        
        st.markdown("---")
        st.markdown('<h2 class="section-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0:
                st.markdown("""
                <div class="prediction-box stay-prediction">
                    <h1>✅ STAY</h1>
                    <p>Employee is likely to <strong>STAY</strong> with the company</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-box leave-prediction">
                    <h1>⚠️ LEAVE</h1>
                    <p>Employee is likely to <strong>LEAVE</strong> the company</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Prediction Probabilities")
            
            st.markdown(f"**🟢 Probability of Staying:** {prob_stay:.1f}%")
            st.markdown(f"""
            <div class="progress-bar-container">
                <div class="progress-bar-green" style="width: {prob_stay}%;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**🔴 Probability of Leaving:** {prob_leave:.1f}%")
            st.markdown(f"""
            <div class="progress-bar-container">
                <div class="progress-bar-red" style="width: {prob_leave}%;"></div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# BATCH PREDICTION TAB
# ============================================================================
def render_batch_prediction_tab(model):
    st.markdown("---")
    st.markdown('<h2 class="section-header">📊 Batch Employee Prediction</h2>', unsafe_allow_html=True)
    
    with st.expander("📋 Required Columns in Your File (Click to Expand)"):
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(245, 158, 11, 0.15)); padding: 1rem; border-radius: 12px; margin-bottom: 1rem; border: 1px solid rgba(251, 191, 36, 0.3);">
            <p style="color: #fcd34d; font-weight: 600;">Your uploaded file <strong>must contain</strong> these columns with <strong>exact names</strong>:</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            for feature in BEST_FEATURES[:3]:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #8b5cf6;">
                    <strong style="color: #a78bfa;">{feature}</strong><br/>
                    <small style="color: rgba(255,255,255,0.6);">{FEATURE_DESCRIPTIONS[feature]}</small>
                </div>
                """, unsafe_allow_html=True)
        with col2:
            for feature in BEST_FEATURES[3:]:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #8b5cf6;">
                    <strong style="color: #a78bfa;">{feature}</strong><br/>
                    <small style="color: rgba(255,255,255,0.6);">{FEATURE_DESCRIPTIONS[feature]}</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.info("💡 **Tip:** Your file can contain additional columns. They will be preserved in the output but won't be used for prediction.")
    
    st.markdown("---")
    
    st.markdown("### 📁 Upload Your Data")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="settings-card">
            <h4>⚙️ File Settings</h4>
        </div>
        """, unsafe_allow_html=True)
        
        file_format = st.selectbox(
            "Select file format",
            options=["CSV", "Excel (.xlsx)"],
            index=0,
            help="Choose the format of your data file",
            key="file_format"
        )
    
    with col2:
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
        
        label_option = st.selectbox(
            "Select prediction labels",
            options=["Leave / Stay", "Yes / No", "Churn / Not Churn", "1 / 0", "True / False", "Custom"],
            index=0,
            help="Choose how predictions should be labeled",
            key="label_option"
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
            help="Add columns showing the probability (%) of staying and leaving",
            key="include_probabilities"
        )
        
        if include_probabilities:
            st.success("✅ Probability columns will be added")
        else:
            st.info("ℹ️ Only prediction label column will be added")
    
    with col2:
        st.markdown("""
        <div class="settings-card">
            <h4>⚠️ High Risk Filter</h4>
        </div>
        """, unsafe_allow_html=True)
        
        include_high_risk_download = st.checkbox(
            "Enable high-risk employees download",
            value=True,
            help="Provide a separate download for employees with >50% probability of leaving",
            key="include_high_risk"
        )
        
        if include_high_risk_download:
            st.success("✅ High-risk download will be available")
        else:
            st.info("ℹ️ Only full results download will be available")
    
    if uploaded_file is not None:
        st.markdown("---")
        st.markdown("### 📄 Uploaded Data Preview")
        
        try:
            if file_format == "CSV":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="stats-card">
                    <h4>📊 Total Rows</h4>
                    <div class="number">{len(df):,}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="stats-card">
                    <h4>📋 Total Columns</h4>
                    <div class="number">{len(df.columns)}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                available_features = [col for col in BEST_FEATURES if col in df.columns]
                st.markdown(f"""
                <div class="stats-card">
                    <h4>✅ Required Cols Found</h4>
                    <div class="number">{len(available_features)}/{len(BEST_FEATURES)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.dataframe(df.head(10), use_container_width=True)
            
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
                
                with st.expander("🔍 View columns being used for prediction"):
                    for feature in BEST_FEATURES:
                        sample_values = df[feature].head(3).tolist()
                        st.write(f"• **{feature}**: Sample values → {sample_values}")
                
                st.markdown("---")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    batch_predict_button = st.button(
                        "🔮 GENERATE BATCH PREDICTIONS",
                        use_container_width=True,
                        key="batch_predict"
                    )
                
                if batch_predict_button:
                    with st.spinner("🔄 Processing predictions..."):
                        input_features = df[BEST_FEATURES].copy()
                        
                        predictions = model.predict(input_features)
                        prediction_probabilities = model.predict_proba(input_features)
                        
                        result_df = df.copy()
                        
                        result_df[prediction_column_name] = [prediction_labels[p] for p in predictions]
                        
                        if include_probabilities:
                            result_df[f"{prediction_column_name}_Probability_Stay"] = (prediction_probabilities[:, 0] * 100).round(2)
                            result_df[f"{prediction_column_name}_Probability_Leave"] = (prediction_probabilities[:, 1] * 100).round(2)
                    
                    st.success("✅ Predictions generated successfully!")
                    
                    st.markdown("---")
                    st.markdown('<h2 class="section-header">📊 Prediction Results</h2>', unsafe_allow_html=True)
                    
                    leaving_count = sum(predictions == 1)
                    staying_count = sum(predictions == 0)
                    leaving_percentage = (leaving_count / len(predictions)) * 100
                    staying_percentage = (staying_count / len(predictions)) * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="stats-card">
                            <h4>👥 Total Employees</h4>
                            <div class="number">{len(predictions):,}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="stats-card" style="border-top-color: #ef4444;">
                            <h4>🚪 Predicted to Leave</h4>
                            <div class="number" style="background: linear-gradient(135deg, #ef4444, #f87171); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{leaving_count:,}</div>
                            <p>({leaving_percentage:.1f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="stats-card" style="border-top-color: #10b981;">
                            <h4>✅ Predicted to Stay</h4>
                            <div class="number" style="background: linear-gradient(135deg, #10b981, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{staying_count:,}</div>
                            <p>({staying_percentage:.1f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        avg_leave_prob = prediction_probabilities[:, 1].mean() * 100
                        st.markdown(f"""
                        <div class="stats-card" style="border-top-color: #fbbf24;">
                            <h4>📈 Avg. Leave Prob</h4>
                            <div class="number" style="background: linear-gradient(135deg, #fbbf24, #f59e0b); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{avg_leave_prob:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("#### 📈 Turnover Distribution")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**🟢 Staying:** {staying_percentage:.1f}%")
                        st.markdown(f"""
                        <div class="progress-bar-container">
                            <div class="progress-bar-green" style="width: {staying_percentage}%;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"**🔴 Leaving:** {leaving_percentage:.1f}%")
                        st.markdown(f"""
                        <div class="progress-bar-container">
                            <div class="progress-bar-red" style="width: {leaving_percentage}%;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                    
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
                            high_risk_df = result_df[prediction_probabilities[:, 1] > 0.5]
                            if len(high_risk_df) > 0:
                                high_risk_csv = convert_df_to_csv(high_risk_df)
                                st.download_button(
                                    label=f"⚠️ High Risk Only ({len(high_risk_df)})",
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
    # Stunning Header
    st.markdown('<h1 class="main-header">Employee Turnover Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">🚀 AI-Powered Workforce Analytics 🚀</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model_from_huggingface()
    
    if model is None:
        st.error("❌ Failed to load model. Please check the Hugging Face repository.")
        st.info(f"Repository: https://huggingface.co/{HF_REPO_ID}")
        return
    
    # Create Tabs
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
