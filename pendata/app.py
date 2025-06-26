# File: app.py
# Kode ini memenuhi ketentuan "deployment terhadap model terbaik tersebut"

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Fungsi untuk memuat model dengan cache agar lebih efisien
@st.cache_resource
def load_model():
    """Fungsi ini memuat pipeline model yang telah disimpan."""
    pipeline = joblib.load('horse_colic_pipeline.pkl')
    return pipeline

# Memuat pipeline model
pipeline = load_model()

# --- Antarmuka Aplikasi Web ---

st.set_page_config(page_title="Prediksi Kolik Kuda", page_icon="🐴", layout="wide")
st.title('🐴 Aplikasi Prediksi Kelangsungan Hidup Kuda')
st.write("""
Aplikasi ini adalah deployment dari model machine learning **Decision Tree**. 
Masukkan data klinis kuda untuk memprediksi apakah kuda tersebut akan selamat atau tidak.
""")

# Membuat kolom untuk tata letak yang lebih rapi
col1, col2 = st.columns(2)

# Opsi untuk dropdown menu agar user-friendly
surgery_options = {1.0: 'Ya, pernah dioperasi', 2.0: 'Tidak pernah dioperasi'}
age_options = {1.0: 'Dewasa (> 6 bulan)', 2.0: 'Muda (< 6 bulan)'}
pain_options = {
    1.0: 'Waspada, tidak sakit', 2.0: 'Depresi', 3.0: 'Nyeri ringan intermiten',
    4.0: 'Nyeri parah intermiten', 5.0: 'Nyeri parah berkelanjutan'
}

# Kolom input pertama
with col1:
    st.header("Data Klinis Utama")
    pulse = st.number_input('Denyut Nadi (pulse)', min_value=30, max_value=200, value=60, help="Denyut jantung per menit (normal: 30-40).")
    rectal_temp = st.number_input('Suhu Rektal (rectal_temperature)', min_value=35.0, max_value=42.0, value=38.0, step=0.1, help="Dalam Celcius (normal: 37.8°C).")
    respiratory_rate = st.number_input('Laju Pernapasan (respiratory_rate)', min_value=8, max_value=100, value=24, help="Napas per menit (normal: 8-10).")
    packed_cell_volume = st.number_input('Volume Sel Darah (packed_cell_volume)', min_value=20.0, max_value=80.0, value=45.0, step=0.1, help="Persentase sel darah merah (normal: 30-50%).")

# Kolom input kedua
with col2:
    st.header("Observasi Kondisi")
    pain_label = st.selectbox('Tingkat Rasa Sakit (pain)', options=list(pain_options.values()))
    surgery_label = st.selectbox('Pernah Dioperasi? (surgery)', options=list(surgery_options.values()))
    age_label = st.selectbox('Usia (age)', options=list(age_options.values()))

# Tombol Prediksi
if st.button('Lakukan Prediksi', type="primary", use_container_width=True):
    # Mengonversi input dari label teks ke nilai numerik yang dipahami model
    pain = [k for k, v in pain_options.items() if v == pain_label][0]
    surgery = [k for k, v in surgery_options.items() if v == surgery_label][0]
    age = [k for k, v in age_options.items() if v == age_label][0]
    
    # Membuat DataFrame dari input pengguna
    input_data = pd.DataFrame({
        'rectal_temperature': [rectal_temp], 'pulse': [pulse], 'respiratory_rate': [respiratory_rate],
        'packed_cell_volume': [packed_cell_volume], 'total_protein': [np.nan], 'surgery': [surgery],
        'age': [age], 'pain': [pain], 'temp_of_extremities': [np.nan], 'peripheral_pulse': [np.nan],
        'mucous_membrane': [np.nan], 'capillary_refill_time': [np.nan], 'peristalsis': [np.nan],
        'abdominal_distension': [np.nan], 'nasogastric_tube': [np.nan], 'nasogastric_reflux': [np.nan],
        'rectal_exam_feces': [np.nan], 'abdomen': [np.nan], 'abdomo_appearance': [np.nan],
        'abdomo_protein': [np.nan], 'surgical_lesion': [np.nan], 'lesion_1': [np.nan], 'lesion_2': [np.nan],
        'lesion_3': [np.nan], 'cp_data': [np.nan]
    })
    
    # Melakukan prediksi dan mendapatkan probabilitas
    prediction = pipeline.predict(input_data)[0]
    prediction_proba = pipeline.predict_proba(input_data)[0]
    
    # Menampilkan hasil
    st.subheader('Hasil Prediksi Model:')
    if prediction == 1:
        st.success('**SELAMAT (Lived)**')
        st.write(f"Keyakinan model: **{prediction_proba[1]*100:.2f}%**")
    else:
        st.error('**TIDAK SELAMAT (Died or Euthanized)**')
        st.write(f"Keyakinan model: **{prediction_proba[0]*100:.2f}%**")