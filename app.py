import streamlit as st
import os
import gdown
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# --- Konfigurasi dan Fungsi Bantuan ---
st.set_page_config(page_title="Deteksi Penyakit Mata", layout="wide")

# Fungsi untuk load model dengan cache agar lebih cepat setelah load pertama
@st.cache_resource
def load_keras_model(path):
    """Memuat model Keras dan menyimpannya di cache."""
    return load_model(path)

# --- Download dan Load Model ---
# URL dan path model
model_url = "https://drive.google.com/uc?id=1cziGtuVG3eoNIwd9j3gzg76ACZ18pDqa"
model_path = "model_multiclass.h5" # Nama file tetap sama, meskipun modelnya biner

# Download model jika belum ada
if not os.path.exists(model_path):
    st.info("⏳ Mendownload model dari Google Drive (hanya sekali)...")
    with st.spinner('Proses download sedang berjalan, mohon tunggu...'):
        gdown.download(model_url, model_path, quiet=False)
    st.success("✅ Model berhasil didownload!")

# Muat model Keras dengan penanganan error
try:
    model = load_keras_model(model_path)
except Exception as e:
    st.error(f"Gagal memuat file model: {e}")
    st.stop() # Hentikan eksekusi jika model gagal dimuat

# --- Antarmuka Pengguna (UI) Streamlit ---
st.title("🧠 Deteksi Penyakit Mata dari Citra Retina")
st.markdown("Aplikasi ini menggunakan model klasifikasi biner untuk mendeteksi **Normal** vs **Katarak**.")

# Komponen untuk mengunggah file
uploaded_file = st.file_uploader("📤 Upload Gambar Retina Mata Anda", type=["jpg", "png", "jpeg"])

# =================================================================================
# BAGIAN UTAMA: SEMUA LOGIKA PEMROSESAN GAMBAR DAN PREDIKSI ADA DI SINI.
# Kode ini hanya akan berjalan SETELAH pengguna berhasil mengunggah sebuah file.
# Ini memperbaiki error `NameError`.
# =================================================================================
if uploaded_file is not None:
    # 1. Buka dan tampilkan gambar yang diunggah
    img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Gambar Asli yang Diunggah", use_container_width=True)

    # 2. Pre-processing gambar agar sesuai dengan input yang diharapkan model
    # Ukuran diubah menjadi 224x224 (ukuran yang umum dan terbukti berhasil sebelumnya)
    target_size = (150, 150) 
    img_resized = img.resize(target_size)
    
    # Konversi gambar ke array NumPy dan normalisasi piksel
    img_array = np.array(img_resized) / 255.0
    
    # Tambahkan dimensi batch (dari (150, 150, 3) menjadi (1, 150, 150, 3))
    img_array = np.expand_dims(img_array, axis=0)

    # 3. Lakukan prediksi dengan model
    prediction_value = model.predict(img_array)[0][0]

    # 4. Tentukan label berdasarkan output model
    # LOGIKA INI SUDAH DIPERBAIKI untuk mengatasi hasil yang terbalik.
    # Asumsi: Model dilatih dengan 'Katarak' sebagai 0 dan 'Normal' sebagai 1.
    if prediction_value < 0.5:
        # Nilai rendah (< 0.5) berarti Katarak
        predicted_label = "Katarak"
        confidence = 1 - prediction_value
    else:
        # Nilai tinggi (>= 0.5) berarti Normal
        predicted_label = "Normal"
        confidence = prediction_value

    # 5. Tampilkan hasil prediksi di kolom kedua
    with col2:
        st.subheader("🔍 Hasil Prediksi")
        
        if predicted_label == "Katarak":
            st.error(f"**Status:** Terdeteksi **{predicted_label}**")
        else:
            st.success(f"**Status:** Terdeteksi **{predicted_label}**")
            
        st.metric(label="Tingkat Keyakinan Model", value=f"{confidence:.2%}")
        
        with st.expander("Lihat Detail Teknis Prediksi"):
            st.write(f"Nilai mentah output dari model: **{prediction_value:.4f}**")
            st.info("""
            **Bagaimana cara membaca ini?** Model ini menghasilkan satu angka. Berdasarkan pelatihannya:
            - Nilai mendekati `0.0` mengindikasikan **Katarak**.
            - Nilai mendekati `1.0` mengindikasikan **Normal**.
            """)
