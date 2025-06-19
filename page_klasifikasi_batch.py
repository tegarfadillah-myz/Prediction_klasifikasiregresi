import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Kustomisasi CSS Injeksi untuk Halaman Spesifik ini ---
st.markdown("""
<style>
/* Header utama di halaman ini */
h2 {
    color: #4CAF50; /* Hijau cerah untuk judul halaman */
    font-family: 'Segoe UI', sans-serif;
    text-align: center;
    margin-bottom: 30px;
    font-size: 2.3em;
    font-weight: 600;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
}

/* Teks intro di bawah header */
.stMarkdown p {
    font-size: 1.05em;
    color: #555555;
    text-align: center;
    margin-bottom: 25px;
}

/* Alert Boxes (Info, Warning, Success, Error) */
.stAlert {
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 25px;
    box-shadow: 3px 3px 10px rgba(0,0,0,0.1);
    font-size: 1.1em;
}
.stAlert.info {
    background-color: #e0f7fa;
    color: #008C99;
    border-left: 6px solid #00BCD4;
}
.stAlert.success {
    background-color: #e8f5e9;
    color: #388E3C;
    border-left: 6px solid #4CAF50;
}
.stAlert.warning {
    background-color: #fff8e1;
    color: #F57F17;
    border-left: 6px solid #FFC107;
}
.stAlert.error {
    background-color: #ffebee;
    color: #D32F2F;
    border-left: 6px solid #F44336;
}

/* Styling untuk file uploader */
.stFileUploader label {
    font-weight: 600;
    color: #34495E;
    margin-bottom: 8px;
    display: block;
    font-size: 1.05em;
}
.stFileUploader [data-testid="stFileUploaderDropzone"] {
    border: 3px dashed #3498DB; /* primaryColor untuk border dashed */
    border-radius: 15px; /* Lebih bulat */
    padding: 30px; /* Padding lebih besar */
    text-align: center;
    background-color: #F8F8F8; /* backgroundColor */
    transition: background-color 0.3s ease, border-color 0.3s ease;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.08);
    cursor: pointer;
}
.stFileUploader [data-testid="stFileUploaderDropzone"]:hover {
    background-color: #E6EEF6; /* secondaryBackgroundColor saat hover */
    border-color: #2980B9; /* Biru lebih gelap */
}
.stFileUploader [data-testid="stFileUploadDropzoneInstructions"] {
    color: #555555;
    font-size: 1.1em;
}
.stFileUploader [data-testid="stFileUploaderUploadAnimation"] {
    color: #28a745; /* Warna hijau untuk animasi sukses */
}


/* Garis pemisah */
hr {
    border-top: 2px solid #E0E0E0;
    margin: 45px 0;
    opacity: 0.7;
}

/* Styling untuk Checkbox Model */
.stCheckbox div[data-testid="stCheckbox"] {
    margin-bottom: 10px;
    font-size: 1.05em;
}
.stCheckbox label {
    cursor: pointer;
    transition: color 0.2s;
}
.stCheckbox label:hover {
    color: #3498DB;
}


/* Tombol Prediksi */
.stButton button {
    background-color: #3498DB; /* primaryColor */
    color: white;
    border-radius: 15px;
    padding: 15px 30px;
    font-size: 1.2em;
    font-weight: bold;
    border: none;
    transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 5px 5px 15px rgba(0,0,0,0.2);
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.stButton button:hover {
    background-color: #2980B9;
    transform: translateY(-3px);
    box-shadow: 7px 7px 20px rgba(0,0,0,0.3);
}
.stButton button:active {
    transform: translateY(0);
    box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
}

/* Styling untuk expander (format CSV) */
.stExpander details {
    background-color: #F0F8FF; /* Biru sangat muda */
    border: 1px solid #BDD9ED;
    border-radius: 12px;
    padding: 15px 20px;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 30px;
}
.stExpander details summary {
    font-weight: bold;
    color: #3498DB; /* primaryColor untuk judul expander */
    font-size: 1.1em;
    cursor: pointer;
    transition: color 0.2s;
}
.stExpander details summary:hover {
    color: #2980B9;
}
.stExpander div[data-testid="stExpanderContents"] {
    padding-top: 15px;
}
.stExpander code { /* Untuk kode CSV di dalam expander */
    background-color: #E6EEF6; /* secondaryBackgroundColor */
    border-radius: 8px;
    padding: 12px;
    display: block;
    overflow-x: auto;
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
    font-size: 0.95em;
    color: #333333;
    border: 1px dashed #AABECF;
}

/* Styling untuk dataframe */
.stDataFrame {
    border: 1px solid #dee2e6;
    border-radius: 12px; /* Lebih membulat */
    box-shadow: 3px 3px 15px rgba(0,0,0,0.1);
    overflow: hidden;
    margin-top: 30px;
}
.stDataFrame th { /* Header kolom dataframe */
    background-color: #3498DB; /* primaryColor */
    color: white;
    font-weight: bold;
    padding: 12px 15px;
    text-align: left;
}
.stDataFrame td { /* Sel-sel data dataframe */
    padding: 10px 15px;
    border-bottom: 1px solid #e9ecef;
    color: #333333;
}
.stDataFrame tbody tr:nth-child(even) { /* Warna baris genap */
    background-color: #FBFBFB;
}
.stDataFrame tbody tr:hover { /* Efek hover pada baris dataframe */
    background-color: #F0F8FF; /* Biru sangat muda saat hover */
    color: #2C3E50;
}


/* Styling untuk tombol download */
.stDownloadButton button {
    background-color: #28A745; /* Warna hijau untuk download */
    color: white;
    border-radius: 15px;
    padding: 15px 30px;
    font-size: 1.2em;
    font-weight: bold;
    border: none;
    transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 5px 5px 15px rgba(0,0,0,0.2);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 30px;
}
.stDownloadButton button:hover {
    background-color: #218838;
    transform: translateY(-3px);
    box-shadow: 7px 7px 20px rgba(0,0,0,0.3);
}

</style>
""", unsafe_allow_html=True)
# --- Akhir Kustomisasi CSS ---


def get_severity_label(prediction):
    if prediction == 1: return "Fatal (Fatal)"
    elif prediction == 2: return "Serious (Serius)"
    elif prediction == 3: return "Slight (Ringan)"
    else: return f"Unknown ({prediction})"

def show_page():
    st.header("Prediksi Batch Keparahan Kecelakaan")
    st.write("Unggah file CSV Anda di sini untuk mendapatkan prediksi keparahan untuk setiap insiden secara massal. Pastikan format kolom sesuai petunjuk.")

    feature_order = ['Number_of_Vehicles', 'Number_of_Casualties', 'Day_of_Week', 'Road_Type', 'Speed_limit', 'Junction_Control', 'Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions', 'Urban_or_Rural_Area', 'Time_Category']
    
    with st.expander("❓ Klik untuk melihat format CSV yang diharapkan"):
        st.write("File CSV Anda harus memiliki semua kolom berikut, dalam urutan ini:")
        st.code(f"{', '.join(feature_order)}")
        st.write("Pastikan nilai-nilai fitur yang di-encode sudah dalam format numerik yang benar.")

    uploaded_file = st.file_uploader("⬆️ Pilih file CSV untuk diunggah", type=["csv"], key="a_b_uploader")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### 📄 Data yang Diunggah (5 baris pertama):")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Gagal memuat file CSV: {e}. Pastikan format file benar dan tidak rusak.")
            return

        st.markdown("---")
        st.write("Pilih model yang ingin digunakan untuk prediksi:")
        
        col_cb1, col_cb2, col_cb3, col_cb4 = st.columns(4)
        with col_cb1:
            use_dt = st.checkbox("Decision Tree", value=True, key="a_b_dt")
        with col_cb2:
            use_knn = st.checkbox("K-Nearest Neighbors", key="a_b_knn")
        with col_cb3:
            use_nn = st.checkbox("Neural Network", key="a_b_nn")
        with col_cb4:
            use_svm = st.checkbox("Support Vector Machine", key="a_b_svm")
        
        if st.button("PREDIKSI UNTUK FILE CSV", type="primary", use_container_width=True, key="a_b_btn"):
            if not any([use_dt, use_knn, use_nn, use_svm]):
                st.warning("⚠️ Silakan pilih setidaknya satu model untuk melanjutkan prediksi.")
                return

            try:
                # Periksa apakah semua kolom yang diperlukan ada dalam dataframe
                missing_cols = [col for col in feature_order if col not in df.columns]
                if missing_cols:
                    st.error(f"❌ Error: Kolom berikut tidak ditemukan di file CSV Anda: **{', '.join(missing_cols)}**. Mohon periksa format CSV Anda.")
                    return

                X = df[feature_order]
                results_df = df.copy()
                st.markdown("---")
                st.header("📈 Hasil Prediksi Batch:")

                def run_batch_prediction(model_name, model_file, column_name):
                    try:
                        if not os.path.exists(model_file):
                            st.error(f"❌ Model {model_name} ('{model_file}') tidak ditemukan. Pastikan path sudah benar.")
                            return

                        model = joblib.load(model_file)
                        predictions = model.predict(X)
                        results_df[column_name] = [get_severity_label(p) for p in predictions]
                    except Exception as e:
                        st.error(f"Terjadi kesalahan pada model **{model_name}**: {e}")
                        st.exception(e) # Tambahkan ini untuk debugging detail
                
                # Menjalankan prediksi hanya untuk model yang dipilih
                if use_dt: run_batch_prediction("Decision Tree","models/modelJb_DecisionTree_klasifikasireal.joblib", "DT_Prediction")
                if use_knn: run_batch_prediction("K-NN", "models/modelJb_ModelKNN_klasifikasi.joblib", "KNN_Prediction")
                if use_nn: run_batch_prediction("NN", "models/modelJb_nn_klasifikasireal.joblib", "NN_Prediction")
                if use_svm: run_batch_prediction("SVM", "models/modelJb_ModelSVM_klasifikasireal.joblib", "SVM_Prediction")

                st.dataframe(results_df, use_container_width=True)
                
                # Opsi download hasil
                csv_output = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Unduh Hasil Prediksi (CSV)",
                    data=csv_output,
                    file_name="hasil_prediksi_batch.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="secondary"
                )

            except KeyError as e:
                st.error(f"❌ Kolom yang diperlukan tidak ditemukan di file CSV Anda: **{e}**. Mohon periksa format CSV Anda.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat pemrosesan batch: {e}")
                st.exception(e) # Tambahkan ini untuk debugging detail