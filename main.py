import streamlit as st

# Ini HARUS jadi perintah Streamlit PERTAMA di script!
st.set_page_config(
    page_title="🤖 AI Prediction Hub - Advanced Analytics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Advanced CSS dengan Animasi dan Efek Modern ---
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --danger-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    --dark-gradient: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    --glass-bg: rgba(255, 255, 255, 0.25);
    --glass-border: rgba(255, 255, 255, 0.18);
    --shadow-light: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    --shadow-heavy: 0 15px 35px rgba(0, 0, 0, 0.1), 0 5px 15px rgba(0, 0, 0, 0.08);
}

/* Global Styling */
* {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    color: #2c3e50;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Main Content Area */
.stApp {
    background: transparent;
}

[data-testid="stAppViewContainer"] {
    background: transparent;
}

[data-testid="stMain"] {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    margin: 20px;
    box-shadow: var(--shadow-heavy);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Enhanced Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 5px 0 25px rgba(0, 0, 0, 0.1);
}

[data-testid="stSidebarContent"] {
    padding: 2rem 1.5rem;
}

/* Sidebar Title dengan Efek Glow */
[data-testid="stSidebarContent"] h1 {
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.8rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 2rem;
    padding: 1rem;
    border-radius: 15px;
    background-color: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    position: relative;
    overflow: hidden;
}

[data-testid="stSidebarContent"] h1::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    transform: rotate(45deg);
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
    100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
}

/* Enhanced Selectbox */
[data-testid="stSidebarContent"] [data-testid="stSelectbox"] label {
    font-weight: 600;
    color: #2c3e50;
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
    display: block;
}

[data-testid="stSidebarContent"] [data-testid="stSelectbox"] > div > div {
    background: rgba(255, 255, 255, 0.9);
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

[data-testid="stSidebarContent"] [data-testid="stSelectbox"] > div > div:hover {
    border-color: #667eea;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    transform: translateY(-2px);
}

/* Animated Radio Buttons */
[data-testid="stSidebarContent"] [data-testid="stRadio"] label {
    font-weight: 500;
    color: #2c3e50;
    font-size: 1rem;
    margin-bottom: 0.8rem;
}

[data-testid="stSidebarContent"] [data-testid="stRadio"] div[role="radiogroup"] > label > div {
    background: rgba(255, 255, 255, 0.8);
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0;
    backdrop-filter: blur(10px);
    cursor: pointer;
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

[data-testid="stSidebarContent"] [data-testid="stRadio"] div[role="radiogroup"] > label > div::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
    transition: left 0.5s;
}

[data-testid="stSidebarContent"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:hover::before {
    left: 100%;
}

[data-testid="stSidebarContent"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    border-color: #667eea;
}

[data-testid="stSidebarContent"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) > div {
    background: var(--primary-gradient);
    color: white;
    font-weight: 600;
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
    border-color: transparent;
}

/* Main Title dengan Efek 3D */
[data-testid="stVerticalBlock"] h1 {
    font-family: 'Inter', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    text-align: center;
    margin: 3rem 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    position: relative;
    animation: titleFloat 6s ease-in-out infinite;
}

@keyframes titleFloat {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

[data-testid="stVerticalBlock"] h1::after {
    content: '';
    position: absolute;
    bottom: -10px;
    left: 50%;
    transform: translateX(-50%);
    width: 100px;
    height: 4px;
    background: var(--primary-gradient);
    border-radius: 2px;
    animation: lineExpand 2s ease-out;
}

@keyframes lineExpand {
    0% { width: 0; }
    100% { width: 100px; }
}

/* Content Container Enhancement */
[data-testid="stVerticalBlock"] {
    padding: 2rem 3rem;
    position: relative;
}

/* Floating Particles Animation */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
        radial-gradient(circle at 40% 40%, rgba(120, 200, 255, 0.3) 0%, transparent 50%);
    animation: particleFloat 20s ease-in-out infinite;
    pointer-events: none;
    z-index: -1;
}

@keyframes particleFloat {
    0%, 100% { transform: translateY(0) rotate(0deg); }
    33% { transform: translateY(-20px) rotate(120deg); }
    66% { transform: translateY(10px) rotate(240deg); }
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 12px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: var(--primary-gradient);
    border-radius: 10px;
    border: 2px solid rgba(255, 255, 255, 0.2);
}

::-webkit-scrollbar-thumb:hover {
    background: var(--secondary-gradient);
}

/* Responsive Design */
@media (max-width: 768px) {
    [data-testid="stVerticalBlock"] h1 {
        font-size: 2.5rem;
    }
    
    [data-testid="stVerticalBlock"] {
        padding: 1rem 1.5rem;
    }
    
    [data-testid="stMain"] {
        margin: 10px;
        border-radius: 15px;
    }
}

/* Hide Streamlit Branding */
.stApp header {
    background-color: transparent;
}

.stApp [data-testid="stToolbar"] {
    display: none;
}

footer {
    display: none;
}

/* Loading Animation */
.stSpinner {
    border: 4px solid rgba(255, 255, 255, 0.3);
    border-top: 4px solid #667eea;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

</style>
""")

# Impor setiap fungsi 'show_page' dari file halamannya masing-masing
import page_regresi_single
import page_regresi_batch
import page_klasifikasi_single
import page_klasifikasi_batch

# --- MENU NAVIGASI DI SIDEBAR ---
st.sidebar.title("🎯 Navigation Hub")

# Pilihan Tingkat 1: Memilih Proyek (Regresi atau Klasifikasi)
project_choice = st.sidebar.selectbox(
    "🎮 Select AI Project:",
    [
        "🎯 Valorant Performance Prediction (Regression)", 
        "🚗 Accident Severity Prediction (Classification)"
    ],
    key="main_project_choice"
)

# Pilihan Tingkat 2: Berdasarkan Proyek yang Dipilih, tampilkan mode yang relevan
if project_choice == "🎯 Valorant Performance Prediction (Regression)":
    
    st.title("🎯 Valorant Performance Analytics")
    
    mode_choice = st.sidebar.radio(
        "🔧 Choose Prediction Mode:",
        ["🎮 Single Match Prediction", "📊 Batch File Analysis"],
        key="valorant_mode"
    )
    
    if mode_choice == "🎮 Single Match Prediction":
        page_regresi_single.show_page()
    elif mode_choice == "📊 Batch File Analysis":
        page_regresi_batch.show_page()

elif project_choice == "🚗 Accident Severity Prediction (Classification)":

    st.title("🚗 Smart Traffic Safety Analytics")

    mode_choice = st.sidebar.radio(
        "🔧 Choose Analysis Mode:",
        ["🔍 Single Incident Analysis", "📈 Batch Data Processing"],
        key="accident_mode"
    )
    
    if mode_choice == "🔍 Single Incident Analysis":
        page_klasifikasi_single.show_page()
    elif mode_choice == "📈 Batch Data Processing":
        page_klasifikasi_batch.show_page()