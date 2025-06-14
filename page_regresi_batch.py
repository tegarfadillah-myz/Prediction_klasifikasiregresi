import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

AGENT_LIST = sorted(['Jett', 'Sage', 'Omen', 'Sova', 'Raze', 'Killjoy', 'Cypher', 'Breach', 'Reyna', 'Viper', 'Phoenix', 'Brimstone', 'Skye', 'Yoru', 'Astra', 'Kayo', 'Chamber', 'Neon', 'Fade', 'Harbor', 'Gekko'])
MAP_LIST = sorted(['Ascent', 'Bind', 'Haven', 'Split', 'Icebox', 'Breeze', 'Fracture', 'Pearl', 'Lotus'])

def show_page():
    st.header("Prediksi Average Combat Score (ACS) secara Batch")
    st.write("Unggah file CSV dengan statistik permainan untuk mendapatkan prediksi ACS untuk setiap baris.")

    with st.expander("Klik untuk melihat format CSV yang diharapkan"):
        st.write("File CSV Anda harus memiliki kolom-kolom berikut: `map,agent,k,d,a,adr,kast,hs,fk,fd`")
    
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"], key="v_b_uploader")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### 📄 Data yang Diunggah (5 baris pertama)", df.head())
            original_df = df.copy()
        except Exception as e:
            st.error(f"Gagal memuat file CSV: {e}")
            return

        st.divider()
        st.subheader("Pilih Model untuk Prediksi")
        use_dt = st.checkbox("Decision Tree", key="v_b_dt")
        use_knn = st.checkbox("K-Nearest Neighbors (KNN)", key="v_b_knn")
        use_nn = st.checkbox("Neural Network (NN)", key="v_b_nn")
        use_svm = st.checkbox("SVM", key="v_b_svm")

        if st.button("Prediksi ACS untuk File", type="primary", use_container_width=True, key="v_b_btn"):
            if not any([use_dt, use_knn, use_nn, use_svm]):
                st.warning("Silakan pilih setidaknya satu model.")
                return

            try:
                numerical_features = df[['k', 'd', 'a', 'kast', 'adr', 'hs', 'fk', 'fd']]
                categorical_features = df[['map', 'agent']]
                encoded_features = pd.get_dummies(categorical_features, columns=['map', 'agent'])
                full_map_cols = [f'map_{m}' for m in MAP_LIST]
                full_agent_cols = [f'agent_{ag}' for ag in AGENT_LIST]
                full_encoded_cols = full_map_cols + full_agent_cols
                encoded_features = encoded_features.reindex(columns=full_encoded_cols, fill_value=0)
                final_column_order = list(numerical_features.columns) + full_encoded_cols
                processed_df = pd.concat([numerical_features, encoded_features], axis=1)[final_column_order]
                
                results_df = original_df.copy()
                st.divider()
                st.header("📈 Hasil Prediksi:")

                def run_batch_reg(model_name, model_file, col_name):
                    try:
                        model = joblib.load(model_file)
                        preds = model.predict(processed_df)
                        results_df[col_name] = [int(p) for p in preds]
                    except Exception as e:
                        st.error(f"Error pada model {model_name}: {e}")

                if use_dt: run_batch_reg("DT", "models/modelJb_DecisionTree_regresireal.joblib", 'ACS_Prediction_DT')
                if use_knn: run_batch_reg("KNN", "models/modelJb_knn_regresireal.joblib", 'ACS_Prediction_KNN')
                if use_nn: run_batch_reg("NN", "models/modelJb_nn_regresireal.joblib", 'ACS_Prediction_NN')
                if use_svm: run_batch_reg("SVM", "models/modelJb_Regresibisa_SVM.joblib", 'ACS_Prediction_SVM')
                
                st.dataframe(results_df, use_container_width=True)

            except KeyError as e:
                st.error(f"Kolom yang diperlukan tidak ditemukan di file CSV Anda: {e}.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat pemrosesan: {e}")