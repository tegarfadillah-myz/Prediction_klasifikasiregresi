import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- KONFIGURASI DAN LIST UNTUK INPUT ---
AGENT_LIST = sorted(['Jett', 'Sage', 'Omen', 'Sova', 'Raze', 'Killjoy', 'Cypher',
                     'Breach', 'Reyna', 'Viper', 'Phoenix', 'Brimstone', 'Skye',
                     'Yoru', 'Astra', 'Kayo', 'Chamber', 'Neon', 'Fade', 'Harbor', 'Gekko'])

MAP_LIST = sorted(['Ascent', 'Bind', 'Haven', 'Split', 'Icebox', 'Breeze', 'Fracture', 'Pearl', 'Lotus'])

AGENT_ROLES = {
    'Jett': 'Duelist', 'Raze': 'Duelist', 'Reyna': 'Duelist', 'Phoenix': 'Duelist', 'Yoru': 'Duelist', 'Neon': 'Duelist',
    'Sage': 'Sentinel', 'Killjoy': 'Sentinel', 'Cypher': 'Sentinel', 'Chamber': 'Sentinel',
    'Omen': 'Controller', 'Viper': 'Controller', 'Brimstone': 'Controller', 'Astra': 'Controller', 'Harbor': 'Controller',
    'Sova': 'Initiator', 'Breach': 'Initiator', 'Skye': 'Initiator', 'Kayo': 'Initiator', 'Fade': 'Initiator', 'Gekko': 'Initiator'
}

def show_page():
    # Custom CSS untuk halaman ini
    st.html("""
    <style>
    /* Enhanced Cards */
    .prediction-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 45px 0 rgba(31, 38, 135, 0.5);
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .prediction-card:hover::before {
        left: 100%;
    }
    
    /* Input Groups */
    .input-group {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .input-group:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    /* Animated Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        font-weight: 600;
        color: #2c3e50;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Progress Animation */
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 8px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 5px;
        animation: progressLoad 2s ease-out;
    }
    
    @keyframes progressLoad {
        0% { width: 0%; }
        100% { width: 100%; }
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Model Selection Cards */
    .model-card {
        background: rgba(255, 255, 255, 0.08);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .model-card:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: rgba(102, 126, 234, 0.3);
        transform: translateX(5px);
    }
    
    .model-card.selected {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border-color: rgba(102, 126, 234, 0.5);
        transform: scale(1.02);
    }
    </style>
    """)
    
    # Header dengan animasi
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h1 style="font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;">
            🎯 Valorant ACS Predictor
        </h1>
        <p style="font-size: 1.2rem; color: #666; font-weight: 500;">
            Prediksi Average Combat Score berdasarkan performa permainan Anda
        </p>
        <div style="width: 100px; height: 4px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin: 1rem auto; border-radius: 2px;"></div>
    </div>
    """, unsafe_allow_html=True)

    # --- INPUT SECTION DENGAN DESIGN ENHANCEMENT ---
    st.markdown("### 🎮 **Match Configuration**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown("#### 🗺️ **Map Selection**")
        selected_map = st.selectbox(
            "Choose your battlefield", 
            MAP_LIST, 
            key="v_s_map",
            help="Select the map where the match was played"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown("#### 🦸 **Agent Selection**")
        selected_agent = st.selectbox(
            "Choose your agent", 
            AGENT_LIST, 
            key="v_s_agent",
            help="Select the agent you played"
        )
        if selected_agent:
            role = AGENT_ROLES.get(selected_agent, "Unknown")
            st.markdown(f"**Role:** `{role}`")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚔️ **Combat Statistics**")
    
    # Combat Stats dengan layout yang lebih menarik
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown("#### 💀 **Kill/Death/Assist**")
        k = st.number_input("🎯 Kills", min_value=0, value=15, step=1, key="v_s_k", help="Total kills in the match")
        d = st.number_input("💥 Deaths", min_value=0, value=15, step=1, key="v_s_d", help="Total deaths in the match")
        a = st.number_input("🤝 Assists", min_value=0, value=8, step=1, key="v_s_a", help="Total assists in the match")
        
        # Calculate KDA ratio
        kda = (k + a) / max(d, 1)
        st.metric("KDA Ratio", f"{kda:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown("#### 📊 **Performance Metrics**")
        adr = st.number_input("💥 Average Damage per Round", min_value=0.0, value=150.0, step=0.1, format="%.1f", key="v_s_adr", help="Average damage dealt per round")
        kast = st.number_input("⭐ KAST (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.1, format="%.1f", key="v_s_kast", help="Kill, Assist, Survive, Trade percentage")
        hs = st.number_input("🎯 Headshot (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.1, format="%.1f", key="v_s_hs", help="Headshot percentage")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown("#### ⚡ **First Blood Stats**")
        fk = st.number_input("🔥 First Kills", min_value=0, value=3, step=1, key="v_s_fk", help="First kills in rounds")
        fd = st.number_input("💀 First Deaths", min_value=0, value=3, step=1, key="v_s_fd", help="First deaths in rounds")
        
        # Calculate First Blood ratio
        fb_ratio = fk / max(fd, 1)
        st.metric("FB Ratio", f"{fb_ratio:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🤖 **Model Selection**")
    
    # Enhanced Model Selection
    st.markdown("Choose one or more AI models for prediction:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_dt = st.checkbox("🌳 **Decision Tree**", key="v_s_dt", help="Tree-based algorithm for interpretable predictions")
        use_knn = st.checkbox("🎯 **K-Nearest Neighbors**", key="v_s_knn", help="Instance-based learning algorithm")
    
    with col2:
        use_nn = st.checkbox("🧠 **Neural Network**", key="v_s_nn", help="Deep learning model for complex patterns")
        use_svm = st.checkbox("⚡ **Support Vector Machine**", key="v_s_svm", help="Kernel-based algorithm for regression")

    st.markdown("---")

    # Enhanced Prediction Button
    if st.button("🚀 **Predict ACS Score**", type="primary", use_container_width=True, key="v_s_btn"):
        
        if not any([use_dt, use_knn, use_nn, use_svm]):
            st.error("🚨 Please select at least one prediction model!")
            return
        
        # Data preprocessing
        numerical_data = pd.DataFrame({
            'k': [k], 'd': [d], 'a': [a], 'kast': [kast], 
            'adr': [adr], 'hs': [hs], 'fk': [fk], 'fd': [fd]
        })
        
        map_df = pd.DataFrame(0, index=[0], columns=[f'map_{m}' for m in MAP_LIST])
        agent_df = pd.DataFrame(0, index=[0], columns=[f'agent_{ag}' for ag in AGENT_LIST])
        map_df[f'map_{selected_map}'] = 1
        agent_df[f'agent_{selected_agent}'] = 1
        processed_df = pd.concat([numerical_data, map_df, agent_df], axis=1)

        # Display loading animation
        with st.spinner('🔮 AI models are analyzing your performance...'):
            st.markdown("---")
            st.markdown("## 📈 **Prediction Results**")
            
            predictions = {}
            model_info = {
                "Decision Tree": {"file": "models/modelJb_DecisionTree_regresireal.joblib", "icon": "🌳", "color": "#2ecc71"},
                "KNN": {"file": "models/modelJb_ModelKNN_REGRESI.joblib", "icon": "🎯", "color": "#3498db"},
                "Neural Network": {"file": "models/modelJb_nn_regresireal.joblib", "icon": "🧠", "color": "#9b59b6"},
                "SVM": {"file": "models/modelJb_Regresibisa_SVM.joblib", "icon": "⚡", "color": "#e74c3c"}
            }
            
            selected_models = []
            if use_dt: selected_models.append("Decision Tree")
            if use_knn: selected_models.append("KNN")
            if use_nn: selected_models.append("Neural Network")
            if use_svm: selected_models.append("SVM")
            
            # Create columns for predictions
            cols = st.columns(len(selected_models))
            
            for idx, model_name in enumerate(selected_models):
                with cols[idx]:
                    try:
                        model = joblib.load(model_info[model_name]["file"])
                        prediction = model.predict(processed_df)[0]
                        predictions[model_name] = int(prediction)
                        
                        # Display result in a beautiful card
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{model_info[model_name]["icon"]}</div>
                            <div class="metric-label">{model_name}</div>
                            <div class="metric-value">{int(prediction)}</div>
                            <div style="font-size: 0.9rem; color: #666;">ACS Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except FileNotFoundError:
                        st.error(f"❌ Model file for {model_name} not found!")
                    except Exception as e:
                        st.error(f"❌ Error with {model_name}: {str(e)}")
            
            # Summary and Analysis
            if predictions:
                st.markdown("---")
                avg_prediction = np.mean(list(predictions.values()))
                max_pred = max(predictions.values())
                min_pred = min(predictions.values())
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                        <div class="metric-label">Average</div>
                        <div class="metric-value">{int(avg_prediction)}</div>
                        <div style="font-size: 0.9rem; color: #666;">Consensus Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">⬆️</div>
                        <div class="metric-label">Highest</div>
                        <div class="metric-value">{max_pred}</div>
                        <div style="font-size: 0.9rem; color: #666;">Optimistic</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">⬇️</div>
                        <div class="metric-label">Lowest</div>
                        <div class="metric-value">{min_pred}</div>
                        <div style="font-size: 0.9rem; color: #666;">Conservative</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Performance Interpretation
                st.markdown("### 📝 **Performance Analysis**")
                
                if avg_prediction >= 250:
                    performance_level = "🏆 **Exceptional Performance**"
                    performance_desc = "Outstanding gameplay! You're performing at a professional level."
                    performance_color = "#f1c40f"
                elif avg_prediction >= 200:
                    performance_level = "⭐ **Excellent Performance**"
                    performance_desc = "Great job! You're consistently outperforming opponents."
                    performance_color = "#2ecc71"
                elif avg_prediction >= 150:
                    performance_level = "👍 **Good Performance**"
                    performance_desc = "Solid gameplay. Keep practicing to reach the next level!"
                    performance_color = "#3498db"
                else:
                    performance_level = "📈 **Room for Improvement**"
                    performance_desc = "Keep grinding! Focus on aim practice and game sense."
                    performance_color = "#e74c3c"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); 
                            padding: 2rem; border-radius: 15px; border: 1px solid rgba(255,255,255,0.2);
                            backdrop-filter: blur(10px); margin: 1rem 0;">
                    <h4 style="color: {performance_color}; margin-bottom: 1rem;">{performance_level}</h4>
                    <p style="font-size: 1.1rem; line-height: 1.6; color: #2c3e50;">{performance_desc}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create a comparison chart
                if len(predictions) > 1:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(predictions.keys()),
                            y=list(predictions.values()),
                            marker=dict(
                                color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'][:len(predictions)],
                                line=dict(color='rgba(255,255,255,0.3)', width=2)
                            ),
                            text=[f"{v}" for v in predictions.values()],
                            textposition='auto',
                            textfont=dict(size=14, color='white')
                        )
                    ])
                    
                    fig.update_layout(
                        title="Model Predictions Comparison",
                        title_font=dict(size=20, color='white'),
                        xaxis_title="AI Models",
                        yaxis_title="Predicted ACS",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        showlegend=False,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

# Helper function to run individual predictions (kept for compatibility)
def run_reg_prediction(model_name, model_file, processed_df):
    try:
        model = joblib.load(model_file)
        prediction = model.predict(processed_df)
        return int(prediction[0])
    except FileNotFoundError:
        st.error(f"File model '{model_file}' tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Error pada model {model_name}: {e}")
        return None