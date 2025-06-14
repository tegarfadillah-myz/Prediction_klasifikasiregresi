import streamlit as st

# Impor setiap fungsi 'show_page' dari file halamannya masing-masing
import page_regresi_single
import page_regresi_batch
import page_klasifikasi_single
import page_klasifikasi_batch

# Konfigurasi Halaman Utama
st.set_page_config(
    page_title="Dasbor Prediksi Terpadu",
    page_icon="🤖",
    layout="wide"
)

# --- MENU NAVIGASI DI SIDEBAR ---
st.sidebar.title("Dasbor Navigasi")

# Pilihan Tingkat 1: Memilih Proyek (Regresi atau Klasifikasi)
project_choice = st.sidebar.selectbox(
    "Pilih Proyek Prediksi:",
    [
        "Prediksi Kinerja Valorant (Regresi)", 
        "Prediksi Keparahan Kecelakaan (Klasifikasi)"
    ]
)

# Pilihan Tingkat 2: Berdasarkan Proyek yang Dipilih, tampilkan mode yang relevan
if project_choice == "Prediksi Kinerja Valorant (Regresi)":
    
    st.title("📈 Prediksi Kinerja Valorant (Regresi)")
    
    mode_choice = st.sidebar.radio(
        "Pilih Mode:",
        ["Prediksi Pertandingan Tunggal", "Unggah File Batch"],
        key="valorant_mode" # Key unik untuk widget ini
    )
    
    if mode_choice == "Prediksi Pertandingan Tunggal":
        page_regresi_single.show_page()
    elif mode_choice == "Unggah File Batch":
        page_regresi_batch.show_page()

elif project_choice == "Prediksi Keparahan Kecelakaan (Klasifikasi)":

    st.title("🚗 Prediksi Keparahan Kecelakaan (Klasifikasi)")

    mode_choice = st.sidebar.radio(
        "Pilih Mode:",
        ["Prediksi Tunggal", "Unggah File Batch"],
        key="accident_mode" # Key unik untuk widget ini
    )
    
    if mode_choice == "Prediksi Tunggal":
        page_klasifikasi_single.show_page()
    elif mode_choice == "Unggah File Batch":
        page_klasifikasi_batch.show_page()