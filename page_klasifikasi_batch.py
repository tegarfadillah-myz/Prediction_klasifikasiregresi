import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

def get_severity_label(prediction):
    if prediction == 1: return "Fatal (Fatal)"
    elif prediction == 2: return "Serious (Serius)"
    elif prediction == 3: return "Slight (Ringan)"
    else: return f"Unknown ({prediction})"

def show_page():
    st.header("Prediksi Batch Keparahan Kecelakaan")
    st.write("Unggah file CSV untuk mendapatkan prediksi keparahan untuk setiap baris data.")

    feature_order = ['Number_of_Vehicles', 'Number_of_Casualties', 'Day_of_Week', 'Road_Type', 'Speed_limit', 'Junction_Control', 'Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions', 'Urban_or_Rural_Area', 'Time_Category']
    with st.expander("Klik untuk melihat format CSV yang diharapkan"):
        st.write("File CSV Anda harus memiliki kolom-kolom berikut:")
        st.code(f"{', '.join(feature_order)}")

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"], key="a_b_uploader")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### 📄 Data yang Diunggah (5 baris pertama)", df.head())
        except Exception as e:
            st.error(f"Gagal memuat file CSV: {e}")
            return

        st.markdown("---")
        st.write("Pilih model yang ingin digunakan untuk prediksi:")
        use_dt = st.checkbox("Gunakan Decision Tree", value=True, key="a_b_dt")
        use_knn = st.checkbox("Gunakan K-Nearest Neighbors", key="a_b_knn")
        use_nn = st.checkbox("Gunakan Neural Network", key="a_b_nn")
        use_svm = st.checkbox("Gunakan Support Vector Machine", key="a_b_svm")
        
        if st.button("Prediksi untuk File CSV", type="primary", use_container_width=True, key="a_b_btn"):
            if not any([use_dt, use_knn, use_nn, use_svm]):
                st.warning("Silakan pilih setidaknya satu model.")
                return

            try:
                X = df[feature_order]
                results_df = df.copy()
                st.markdown("---")
                st.header("📈 Hasil Prediksi Batch:")

                def run_batch_prediction(model_name, model_file, column_name):
                    try:
                        model = joblib.load(model_file)
                        predictions = model.predict(X)
                        results_df[column_name] = [get_severity_label(p) for p in predictions]
                    except FileNotFoundError:
                        st.error(f"Model {model_name} ('{model_file}') tidak ditemukan.")
                    except Exception as e:
                        st.error(f"Error pada model {model_name}: {e}")
                
                if use_dt: run_batch_prediction("Decision Tree","models/modelJb_DecisionTree_klasifikasireal.joblib", "DT_Prediction")
                if use_knn: run_batch_prediction("K-NN", "models/modelJb_ModelKNN_klasifikasi.joblib", "KNN_Prediction")
                if use_nn: run_batch_prediction("NN", "models/modelJb_nn_klasifikasireal.joblib", "NN_Prediction")
                if use_svm: run_batch_prediction("SVM", "models/modelJb_ModelSVM_klasifikasireal.joblib", "SVM_Prediction")

                st.dataframe(results_df, use_container_width=True)
            except KeyError as e:
                st.error(f"Kolom yang diperlukan tidak ditemukan di file CSV Anda: {e}.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat pemrosesan: {e}")