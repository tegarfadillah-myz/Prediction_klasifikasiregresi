import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

def show_page():
    st.header("Prediksi Tunggal Keparahan Kecelakaan")
    st.write("Masukkan nilai untuk setiap fitur di bawah ini untuk mendapatkan prediksi.")
    st.info("CATATAN: Masukkan nilai numerik untuk fitur-fitur yang sudah di-LabelEncode.")

    col1, col2 = st.columns(2)
    with col1:
        num_vehicles = st.number_input("Jumlah Kendaraan", min_value=1, value=2, step=1, key="a_s_veh")
        num_casualties = st.number_input("Jumlah Korban", min_value=1, value=1, step=1, key="a_s_cas")
        day_of_week = st.selectbox("Hari dalam Seminggu", options=list(range(1, 8)), format_func=lambda x: f"Hari ke-{x}", index=3, key="a_s_day")
        road_type = st.number_input("Tipe Jalan (encoded)", min_value=0, value=0, step=1, help="Contoh: 0, 1, 2, ...", key="a_s_road")
        speed_limit = st.number_input("Batas Kecepatan", min_value=10, value=30, step=10, key="a_s_speed")
        junction_control = st.number_input("Kontrol Persimpangan (encoded)", min_value=0, value=0, step=1, help="Contoh: 0, 1, 2, ...", key="a_s_junc")
    with col2:
        light_conditions = st.number_input("Kondisi Cahaya (encoded)", min_value=0, value=0, step=1, help="Contoh: 0, 1, 2, ...", key="a_s_light")
        weather_conditions = st.number_input("Kondisi Cuaca (encoded)", min_value=0, value=0, step=1, help="Contoh: 0, 1, 2, ...", key="a_s_weather")
        road_surface = st.number_input("Kondisi Permukaan Jalan (encoded)", min_value=0, value=0, step=1, help="Contoh: 0, 1, 2, ...", key="a_s_surf")
        urban_rural = st.selectbox("Area Perkotaan/Pedesaan", options=[1, 2, 3], format_func=lambda x: f"Area Tipe {x}", index=0, key="a_s_urban")
        time_category = st.selectbox("Kategori Waktu", options=[0, 1, 2, 3], format_func=lambda x: ["Pagi", "Siang", "Sore", "Malam"][x], index=2, key="a_s_time")

    st.markdown("---")
    st.write("Pilih model yang ingin digunakan untuk prediksi:")
    use_dt = st.checkbox("Gunakan Decision Tree", value=True, key="a_s_dt")
    use_knn = st.checkbox("Gunakan K-Nearest Neighbors", key="a_s_knn")
    use_nn = st.checkbox("Gunakan Neural Network", key="a_s_nn")
    use_svm = st.checkbox("Gunakan Support Vector Machine", key="a_s_svm")

    if st.button("Prediksi Sekarang", type="primary", use_container_width=True, key="a_s_btn"):
        feature_order = ['Number_of_Vehicles', 'Number_of_Casualties', 'Day_of_Week', 'Road_Type', 'Speed_limit', 'Junction_Control', 'Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions', 'Urban_or_Rural_Area', 'Time_Category']
        input_list = [num_vehicles, num_casualties, day_of_week, road_type, speed_limit, junction_control, light_conditions, weather_conditions, road_surface, urban_rural, time_category]
        input_df = pd.DataFrame([input_list], columns=feature_order)

        st.markdown("---")
        st.header("📈 Hasil Prediksi:")

        if not any([use_dt, use_knn, use_nn, use_svm]):
            st.warning("Pilih setidaknya satu model.")
            return

        def run_prediction(model_name, model_file):
            try:
                model = joblib.load(model_file)
                pred = model.predict(input_df)
                if pred[0] == 1: label = "Fatal (Fatal)"
                elif pred[0] == 2: label = "Serious (Serius)"
                elif pred[0] == 3: label = "Slight (Ringan)"
                else: label = f"Unknown ({pred[0]})"
                st.subheader(f"Hasil Prediksi Model {model_name}:")
                st.success(f"Tingkat Keparahan Kecelakaan: **{label}**")
            except FileNotFoundError:
                st.error(f"Error: File model '{model_file}' tidak ditemukan.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi dengan {model_name}: {e}")

        if use_dt: run_prediction("Decision Tree", "models/modelJb_DecisionTree_klasifikasireal.joblib")
        if use_knn: run_prediction("K-Nearest Neighbors", "models/modelJb_ModelKNN_klasifikasi.joblib")
        if use_nn: run_prediction("Neural Network (MLP)", "models/modelJb_nn_klasifikasireal.joblib")
        if use_svm: run_prediction("Support Vector Machine", "models/modelJb_ModelSVM_klasifikasireal.joblib")