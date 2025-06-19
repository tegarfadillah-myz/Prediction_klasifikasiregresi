import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- KONFIGURASI DATA ---
SEVERITY_MAPPING = {
    1: {"label": "Fatal", "icon": "💀", "color": "#e74c3c", "description": "Kecelakaan dengan korban meninggal"},
    2: {"label": "Serious", "icon": "🚨", "color": "#f39c12", "description": "Kecelakaan dengan luka serius"},
    3: {"label": "Slight", "icon": "🟡", "color": "#2ecc71", "description": "Kecelakaan dengan luka ringan"}
}

ROAD_TYPES = {
    0: "Roundabout", 1: "One way street", 2: "Dual carriageway", 3: "Single carriageway",
    4: "Slip road", 5: "Unknown"
}

JUNCTION_CONTROLS = {
    0: "Data missing", 1: "Authorised person", 2: "Auto traffic signal", 3: "Stop sign",
    4: "Give way or uncontrolled"
}

LIGHT_CONDITIONS = {
    0: "Daylight", 1: "Darkness - lights lit", 2: "Darkness - lights unlit", 
    3: "Darkness - no lighting", 4: "Darkness - lighting unknown"
}

WEATHER_CONDITIONS = {
    0: "Fine no high winds", 1: "Raining no high winds", 2: "Snowing no high winds",
    3: "Fine + high winds", 4: "Raining + high winds", 5: "Fog or mist", 6: "Other", 7: "Unknown"
}

ROAD_SURFACES = {
    0: "Dry", 1: "Wet or damp", 2: "Snow", 3: "Frost or ice", 4: "Flood over 3cm deep"
}

AREA_TYPES = {
    1: "Urban", 2: "Rural", 3: "Unallocated"
}

TIME_CATEGORIES = {
    0: "Pagi (06:00-12:00)", 1: "Siang (12:00-18:00)", 
    2: "Sore (18:00-00:00)", 3: "Malam (00:00-06:00)"
}

DAYS = {
    1: "Senin", 2: "Selasa", 3: "Rabu", 4: "Kamis", 
    5: "Jumat", 6: "Sabtu", 7: "Minggu"
}

def get_risk_level(predictions):
    """Calculate risk level based on predictions"""
    if not predictions:
        return "Unknown", "#95a5a6"
    
    fatal_count = sum(1 for p in predictions.values() if p == 1)
    serious_count = sum(1 for p in predictions.values() if p == 2)
    slight_count = sum(1 for p in predictions.values() if p == 3)
    
    total = len(predictions)
    
    if total == 0: # Handle case where no models predicted
        return "No Prediction", "#95a5a6"

    if fatal_count >= total * 0.5:
        return "Very High Risk", "#c0392b"
    elif serious_count >= total * 0.5:
        return "High Risk", "#e67e22"
    elif fatal_count + serious_count >= total * 0.5:
        return "Moderate Risk", "#f39c12"
    else:
        return "Low Risk", "#27ae60"

def show_page():
    # Custom CSS untuk halaman klasifikasi
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
    
    /* Severity Cards */
    .severity-card {
        border: 2px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .severity-card.fatal {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.1) 0%, rgba(192, 57, 43, 0.1) 100%);
        border-color: rgba(231, 76, 60, 0.3);
    }
    
    .severity-card.serious {
        background: linear-gradient(135deg, rgba(243, 156, 18, 0.1) 0%, rgba(230, 126, 34, 0.1) 100%);
        border-color: rgba(243, 156, 18, 0.3);
    }
    
    .severity-card.slight {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.1) 0%, rgba(39, 174, 96, 0.1) 100%);
        border-color: rgba(46, 204, 113, 0.3);
    }
    
    .severity-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
    }
    
    .severity-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .severity-label {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .severity-desc {
        font-size: 0.9rem;
        opacity: 0.8;
        line-height: 1.4;
    }
    
    /* Risk Level Display */
    .risk-indicator {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        border: 2px solid rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .risk-indicator::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .risk-level {
        font-size: 2rem;
        font-weight: 800;
        margin: 1rem 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    /* Enhanced Model Cards */
    .model-result-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .model-result-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);
    }
    
    .model-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .model-name {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    
    /* Statistics Cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin: 0.5rem 0;
    }
    
    .stat-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #2c3e50;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Warning/Info boxes */
    .info-box {
        background: rgba(52, 152, 219, 0.1);
        border: 1px solid rgba(52, 152, 219, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .warning-box {
        background: rgba(243, 156, 18, 0.1);
        border: 1px solid rgba(243, 156, 18, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    </style>
    """)
    
    # Header dengan animasi
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h1 style="font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;">
            🚗 Traffic Safety Predictor
        </h1>
        <p style="font-size: 1.2rem; color: #666; font-weight: 500;">
            Prediksi tingkat keparahan kecelakaan berdasarkan kondisi lalu lintas
        </p>
        <div style="width: 100px; height: 4px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin: 1rem auto; border-radius: 2px;"></div>
    </div>
    """, unsafe_allow_html=True)

    # Info box
    st.markdown("""
    <div class="info-box">
        <h4 style="color: #3498db; margin-bottom: 1rem;">📋 Petunjuk Penggunaan</h4>
        <p style="margin: 0; line-height: 1.6;">
            Masukkan informasi detail tentang kondisi kecelakaan untuk mendapatkan prediksi tingkat keparahan. 
            Semua nilai numerik telah di-encode sesuai dengan standar data kecelakaan lalu lintas.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- INPUT SECTION ---
    st.markdown("### 🚦 **Accident Context Information**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown("#### 🚗 **Vehicle & Casualty Info**")
        num_vehicles = st.number_input(
            "🚗 Number of Vehicles", 
            min_value=1, value=2, step=1, key="a_s_veh",
            help="Total number of vehicles involved in the accident"
        )
        num_casualties = st.number_input(
            "🩹 Number of Casualties", 
            min_value=1, value=1, step=1, key="a_s_cas",
            help="Total number of people injured in the accident"
        )
        
        # Calculate severity indicator
        severity_ratio = num_casualties / num_vehicles
        st.metric("Casualty/Vehicle Ratio", f"{severity_ratio:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown("#### 🛣️ **Road Conditions**")
        road_type = st.selectbox(
            "🛣️ Road Type", 
            options=list(ROAD_TYPES.keys()),
            format_func=lambda x: f"{x} - {ROAD_TYPES[x]}",
            key="a_s_road",
            help="Type of road where accident occurred"
        )
        speed_limit = st.number_input(
            "⚡ Speed Limit (mph)", 
            min_value=10, value=30, step=10, key="a_s_speed",
            help="Posted speed limit at accident location"
        )
        junction_control = st.selectbox(
            "🚥 Junction Control", 
            options=list(JUNCTION_CONTROLS.keys()),
            format_func=lambda x: f"{x} - {JUNCTION_CONTROLS[x]}",
            key="a_s_junc",
            help="Type of traffic control at junction"
        )
        road_surface = st.selectbox(
            "🛤️ Road Surface", 
            options=list(ROAD_SURFACES.keys()),
            format_func=lambda x: f"{x} - {ROAD_SURFACES[x]}",
            key="a_s_surf",
            help="Condition of road surface"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown("#### 📅 **Time & Location**")
        day_of_week = st.selectbox(
            "📅 Day of Week", 
            options=list(DAYS.keys()),
            format_func=lambda x: f"{DAYS[x]}",
            index=3, key="a_s_day",
            help="Day when accident occurred"
        )
        time_category = st.selectbox(
            "🕐 Time Category", 
            options=list(TIME_CATEGORIES.keys()),
            format_func=lambda x: TIME_CATEGORIES[x],
            index=2, key="a_s_time",
            help="Time period when accident occurred"
        )
        urban_rural = st.selectbox(
            "🏙️ Area Type", 
            options=list(AREA_TYPES.keys()),
            format_func=lambda x: f"{AREA_TYPES[x]}",
            key="a_s_urban",
            help="Urban or rural area classification"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown("#### 🌤️ **Environmental Conditions**")
        light_conditions = st.selectbox(
            "💡 Light Conditions", 
            options=list(LIGHT_CONDITIONS.keys()),
            format_func=lambda x: f"{x} - {LIGHT_CONDITIONS[x]}",
            key="a_s_light",
            help="Lighting conditions during accident"
        )
        weather_conditions = st.selectbox(
            "🌦️ Weather Conditions", 
            options=list(WEATHER_CONDITIONS.keys()),
            format_func=lambda x: f"{x} - {WEATHER_CONDITIONS[x]}",
            key="a_s_weather",
            help="Weather conditions during accident"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🤖 **AI Model Selection**")
    
    st.markdown("Choose one or more AI models for accident severity prediction:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_dt = st.checkbox("🌳 **Decision Tree**", value=True, key="a_s_dt", help="Tree-based interpretable model")
    with col2:
        use_knn = st.checkbox("🎯 **K-Nearest Neighbors**", key="a_s_knn", help="Instance-based learning model")
    
    col3, col4 = st.columns(2)
    with col3:
        use_nn = st.checkbox("🧠 **Neural Network (MLP)**", key="a_s_nn", help="Multi-layer perceptron deep learning model")
    with col4:
        use_svm = st.checkbox("⚡ **Support Vector Machine**", key="a_s_svm", help="Kernel-based classification model")

    st.markdown("---")

    # Enhanced Prediction Button
    if st.button("🔮 **Analyze Accident Severity**", type="primary", use_container_width=True, key="a_s_btn"):
        
        if not any([use_dt, use_knn, use_nn, use_svm]):
            st.error("🚨 Please select at least one prediction model!")
            return
        
        # Prepare input data
        feature_order = ['Number_of_Vehicles', 'Number_of_Casualties', 'Day_of_Week', 'Road_Type', 
                         'Speed_limit', 'Junction_Control', 'Light_Conditions', 'Weather_Conditions', 
                         'Road_Surface_Conditions', 'Urban_or_Rural_Area', 'Time_Category']
        
        input_list = [num_vehicles, num_casualties, day_of_week, road_type, speed_limit, 
                      junction_control, light_conditions, weather_conditions, road_surface, 
                      urban_rural, time_category]
        
        input_df = pd.DataFrame([input_list], columns=feature_order)

        # Display loading animation
        with st.spinner('🔍 AI models are analyzing accident conditions...'):
            st.markdown("---")
            st.markdown("## 📊 **Severity Analysis Results**")
            
            predictions = {}
            model_info = {
                "Decision Tree": {"file": "models/modelJb_DecisionTree_klasifikasireal.joblib", "icon": "🌳", "color": "#2ecc71"},
                "K-Nearest Neighbors": {"file": "models/modelJb_ModelKNN_klasifikasi.joblib", "icon": "🎯", "color": "#3498db"},
                "Neural Network (MLP)": {"file": "models/modelJb_nn_klasifikasireal.joblib", "icon": "🧠", "color": "#9b59b6"},
                "Support Vector Machine": {"file": "models/modelJb_ModelSVM_klasifikasireal.joblib", "icon": "⚡", "color": "#e74c3c"}
            }
            
            selected_models = []
            if use_dt: selected_models.append("Decision Tree")
            if use_knn: selected_models.append("K-Nearest Neighbors")
            if use_nn: selected_models.append("Neural Network (MLP)")
            if use_svm: selected_models.append("Support Vector Machine")
            
            # Create columns for predictions dynamically
            # If only 1 model selected, use 1 column. If 2, use 2. If 3 or 4, split into 2 rows of 2 columns each.
            num_cols = len(selected_models)
            if num_cols == 0:
                st.warning("No models selected for prediction.")
                return

            # Divide models into two rows if more than 2 models are selected
            if num_cols > 2:
                col_sets = [st.columns(2), st.columns(2)]
                model_idx = 0
                for model_name in selected_models:
                    current_col_set = col_sets[0] if model_idx < 2 else col_sets[1]
                    with current_col_set[model_idx % 2]:
                        try:
                            model = joblib.load(model_info[model_name]["file"])
                            pred = model.predict(input_df)[0]
                            predictions[model_name] = pred
                            
                            severity_info = SEVERITY_MAPPING[pred]
                            severity_class = severity_info["label"].lower()
                            
                            st.markdown(f"""
                            <div class="severity-card {severity_class}">
                                <span class="severity-icon">{model_info[model_name]["icon"]}</span>
                                <div class="model-name" style="color: {model_info[model_name]['color']};">{model_name}</div>
                                <span class="severity-icon">{severity_info["icon"]}</span>
                                <div class="severity-label" style="color: {severity_info['color']};">{severity_info["label"]}</div>
                                <div class="severity-desc">{severity_info["description"]}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        except FileNotFoundError:
                            st.error(f"❌ Model file for {model_name} not found!")
                        except Exception as e:
                            st.error(f"❌ Error with {model_name}: {str(e)}")
                    model_idx += 1
            else: # 1 or 2 models selected, use a single row of columns
                cols_single_row = st.columns(num_cols)
                for idx, model_name in enumerate(selected_models):
                    with cols_single_row[idx]:
                        try:
                            model = joblib.load(model_info[model_name]["file"])
                            pred = model.predict(input_df)[0]
                            predictions[model_name] = pred
                            
                            severity_info = SEVERITY_MAPPING[pred]
                            severity_class = severity_info["label"].lower()
                            
                            st.markdown(f"""
                            <div class="severity-card {severity_class}">
                                <span class="severity-icon">{model_info[model_name]["icon"]}</span>
                                <div class="model-name" style="color: {model_info[model_name]['color']};">{model_name}</div>
                                <span class="severity-icon">{severity_info["icon"]}</span>
                                <div class="severity-label" style="color: {severity_info['color']};">{severity_info["label"]}</div>
                                <div class="severity-desc">{severity_info["description"]}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        except FileNotFoundError:
                            st.error(f"❌ Model file for {model_name} not found!")
                        except Exception as e:
                            st.error(f"❌ Error with {model_name}: {str(e)}")


            # Analysis and Summary
            if predictions:
                st.markdown("---")
                
                # Calculate statistics
                fatal_count = sum(1 for p in predictions.values() if p == 1)
                serious_count = sum(1 for p in predictions.values() if p == 2)
                slight_count = sum(1 for p in predictions.values() if p == 3)
                total_models = len(predictions)
                
                # Risk level assessment
                risk_level, risk_color = get_risk_level(predictions)
                
                # Display risk indicator
                st.markdown(f"""
                <div class="risk-indicator" style="border-color: {risk_color};">
                    <h3 style="color: #2c3e50; margin-bottom: 1rem;">🚨 Overall Risk Assessment</h3>
                    <div class="risk-level" style="color: {risk_color};">{risk_level}</div>
                    <p style="color: #666; font-size: 1.1rem; margin: 0;">
                        Based on {total_models} AI model predictions
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Statistics breakdown
                col1_stat, col2_stat, col3_stat = st.columns(3)
                
                with col1_stat:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">💀</div>
                        <div class="stat-value">{fatal_count}</div>
                        <div class="stat-label">Fatal Predictions</div>
                        <div style="font-size: 0.9rem; color: #666;">{(fatal_count/total_models*100):.1f}% of models</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2_stat:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🚨</div>
                        <div class="stat-value">{serious_count}</div>
                        <div class="stat-label">Serious Predictions</div>
                        <div style="font-size: 0.9rem; color: #666;">{(serious_count/total_models*100):.1f}% of models</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3_stat:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🟡</div>
                        <div class="stat-value">{slight_count}</div>
                        <div class="stat-label">Slight Predictions</div>
                        <div style="font-size: 0.9rem; color: #666;">{(slight_count/total_models*100):.1f}% of models</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Safety recommendations
                st.markdown("### 🛡️ **Safety Recommendations**")
                
                if fatal_count > 0:
                    recommendations = [
                        "🚨 Immediate emergency response required",
                        "🏥 Ensure proper medical facilities are available",
                        "🚧 Implement strict traffic control measures",
                        "📊 Consider road safety improvements at this location"
                    ]
                    recommendation_color = "#e74c3c"
                    recommendation_level = "Critical Safety Alert"
                elif serious_count > 0:
                    recommendations = [
                        "⚠️ Enhanced safety measures recommended",
                        "🚑 Ensure ambulance accessibility",
                        "🔍 Monitor traffic patterns closely",
                        "💡 Consider improved lighting/signage"
                    ]
                    recommendation_color = "#f39c12"
                    recommendation_level = "High Priority Safety"
                else:
                    recommendations = [
                        "✅ Standard safety protocols apply",
                        "👮 Regular patrol monitoring",
                        "📋 Document incident for future reference",
                        "🔄 Continue routine safety checks"
                    ]
                    recommendation_color = "#2ecc71"
                    recommendation_level = "Standard Safety Protocol"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); 
                            padding: 2rem; border-radius: 15px; border: 2px solid {recommendation_color}30;
                            backdrop-filter: blur(10px); margin: 1rem 0;">
                    <h4 style="color: {recommendation_color}; margin-bottom: 1.5rem;">{recommendation_level}</h4>
                    <ul style="list-style: none; padding: 0; margin: 0;">
                        {"".join([f'<li style="padding: 0.5rem 0; font-size: 1.1rem; line-height: 1.6; color: #2c3e50;">{rec}</li>' for rec in recommendations])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Create prediction distribution chart
                if len(predictions) > 1:
                    st.markdown("### 📊 **Prediction Distribution Across Models**")
                    severity_counts = [fatal_count, serious_count, slight_count]
                    severity_labels = ['Fatal', 'Serious', 'Slight']
                    colors = ['#e74c3c', '#f39c12', '#2ecc71']
                    
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=severity_labels,
                            values=severity_counts,
                            hole=0.4,
                            marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
                            textinfo='label+percent+value',
                            textfont=dict(size=14, color='white'),
                            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Prediction Distribution",
                        title_font=dict(size=20, color='white'),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.2,
                            xanchor="center",
                            x=0.5
                        ),
                        height=400, # Adjusted height for better fit
                        margin=dict(t=50, b=50, l=50, r=50) # Added margins
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

                # Detailed model comparison
                if len(predictions) > 1:
                    st.markdown("### 📈 **Model Performance Comparison**")
                    
                    model_names = list(predictions.keys())
                    # Convert numerical predictions to their labels for the DataFrame
                    pred_values = [SEVERITY_MAPPING[pred]["label"] for pred in predictions.values()]
                    
                    comparison_df = pd.DataFrame({
                        'Model': model_names,
                        'Prediction': pred_values,
                        'Severity_Code': list(predictions.values()) # Keep numerical code for sorting/color mapping
                    })

                    # Map severity codes to colors for the bar chart
                    color_map = {1: '#e74c3c', 2: '#f39c12', 3: '#2ecc71'}
                    comparison_df['Color'] = comparison_df['Severity_Code'].map(color_map)

                    fig_bar = px.bar(
                        comparison_df,
                        x='Model',
                        y='Severity_Code',
                        color='Prediction', # Use prediction label for legend
                        color_discrete_map={'Fatal': '#e74c3c', 'Serious': '#f39c12', 'Slight': '#2ecc71'},
                        labels={'Severity_Code': 'Predicted Severity (1=Fatal, 2=Serious, 3=Slight)', 'Model': 'AI Model'},
                        title='Predicted Severity by Each Model',
                        text='Prediction' # Display the label on the bar
                    )

                    fig_bar.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        title_font=dict(size=20, color='white'),
                        xaxis_title_font=dict(size=14, color='white'),
                        yaxis_title_font=dict(size=14, color='white'),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=False, tickvals=[1,2,3], ticktext=['Fatal', 'Serious', 'Slight']),
                        hovermode="x unified"
                    )
                    
                    # Ensure text labels are visible
                    fig_bar.update_traces(textposition='outside', textfont=dict(color='white'))


                    st.plotly_chart(fig_bar, use_container_width=True)
                
                st.markdown("---")
                st.markdown("""
                <div style="text-align: center; padding: 2.5rem; background: rgba(255,255,255,0.1); border-radius: 20px; margin-top: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                    <h3 style="color: #667eea; margin-bottom: 1rem;">✨ Analysis Complete!</h3>
                    <p style="font-size: 1.1rem; color: #BBB;">
                        Prediksi tingkat keparahan kecelakaan telah selesai. Gunakan informasi ini untuk mengambil keputusan yang lebih baik dalam penanganan dan pencegahan kecelakaan.
                    </p>
                    <p style="font-size: 0.9rem; color: #999;">
                        <em>Disclaimer: Hasil prediksi berdasarkan model AI dan data pelatihan. Akurasi dapat bervariasi. Selalu gunakan penilaian profesional dalam situasi nyata.</em>
                    </p>
                </div>
                """, unsafe_allow_html=True)

# Note: Make sure your model files (e.g., 'models/modelJb_DecisionTree_klasifikasireal.joblib') exist in the specified 'models/' directory.
# Also, ensure 'page_regresi_single.py' is the correct name if this code is in a separate file being imported.