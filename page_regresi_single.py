import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- KONFIGURASI DAN LIST UNTUK INPUT ---
AGENT_LIST = sorted(['Jett', 'Sage', 'Omen', 'Sova', 'Raze', 'Killjoy', 'Cypher',
                     'Breach', 'Reyna', 'Viper', 'Phoenix', 'Brimstone', 'Skye',
                     'Yoru', 'Astra', 'Kayo', 'Chamber', 'Neon', 'Fade', 'Harbor', 'Gekko'])

MAP_LIST = sorted(['Ascent', 'Bind', 'Haven', 'Split', 'Icebox', 'Breeze', 'Fracture', 'Pearl', 'Lotus'])

def show_page():
    st.header("Prediksi Average Combat Score (ACS)")
    st.write("Masukkan statistik permainan untuk memprediksi ACS.")
    st.divider()

    # --- INPUT DARI PENGGUNA ---
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Konteks Pertandingan")
            selected_map = st.selectbox("Pilih Map", MAP_LIST, key="v_s_map")
            selected_agent = st.selectbox("Pilih Agent", AGENT_LIST, key="v_s_agent")
            st.divider()
            st.subheader("Statistik Tempur Utama")
            k = st.number_input("Kills (k)", min_value=0, value=15, step=1, key="v_s_k")
            d = st.number_input("Deaths (d)", min_value=0, value=15, step=1, key="v_s_d")
            a = st.number_input("Assists (a)", min_value=0, value=8, step=1, key="v_s_a")

        with col2:
            st.subheader("Statistik Performa & Dampak")
            adr = st.number_input("Average Damage per Round (adr)", min_value=0.0, value=150.0, step=0.1, format="%.1f", key="v_s_adr")
            kast = st.number_input("KAST (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.1, format="%.1f", key="v_s_kast")
            hs = st.number_input("Headshot (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.1, format="%.1f", key="v_s_hs")
            st.divider()
            st.subheader("Statistik Pembuka Ronde")
            fk = st.number_input("First Kills (fk)", min_value=0, value=3, step=1, key="v_s_fk")
            fd = st.number_input("First Deaths (fd)", min_value=0, value=3, step=1, key="v_s_fd")

    st.divider()

    st.subheader("Pilih Model untuk Prediksi")
    use_dt = st.checkbox("Decision Tree", key="v_s_dt")
    use_knn = st.checkbox("K-Nearest Neighbors (KNN)", key="v_s_knn")
    use_nn = st.checkbox("Neural Network (NN)", key="v_s_nn")
    use_svm = st.checkbox("SVM", key="v_s_svm")

    if st.button("Prediksi ACS", type="primary", use_container_width=True, key="v_s_btn"):
        numerical_data = pd.DataFrame({'k': [k], 'd': [d], 'a': [a], 'kast': [kast], 'adr': [adr], 'hs': [hs], 'fk': [fk], 'fd': [fd]})
        map_df = pd.DataFrame(0, index=[0], columns=[f'map_{m}' for m in MAP_LIST])
        agent_df = pd.DataFrame(0, index=[0], columns=[f'agent_{ag}' for ag in AGENT_LIST])
        map_df[f'map_{selected_map}'] = 1
        agent_df[f'agent_{selected_agent}'] = 1
        processed_df = pd.concat([numerical_data, map_df, agent_df], axis=1)

        st.divider()
        st.header("📈 Hasil Prediksi:")

        if not any([use_dt, use_knn, use_nn, use_svm]):
            st.warning("Pilih setidaknya satu model.")
            return

        def run_reg_prediction(model_name, model_file):
            try:
                model = joblib.load(model_file)
                prediction = model.predict(processed_df)
                st.subheader(f"Prediksi ACS Model {model_name}:")
                st.success(f"**{int(prediction[0])}**")
            except FileNotFoundError:
                st.error(f"File model '{model_file}' tidak ditemukan.")
            except Exception as e:
                st.error(f"Error pada model {model_name}: {e}")

        if use_dt: run_reg_prediction("Decision Tree", "models/modelJb_DecisionTree_regresireal.joblib")
        if use_knn: run_reg_prediction("KNN", "models/modelJb_knn_regresireal.joblib")
        if use_nn: run_reg_prediction("Neural Network", "models/modelJb_nn_regresireal.joblib")
        if use_svm: run_reg_prediction("SVM", "models/modelJb_Regresibisa_SVM.joblib")