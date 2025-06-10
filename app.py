import streamlit as st
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- Step 1: Download model from Google Drive if not exist ---
model_url = "https://drive.google.com/uc?id=1cziGtuVG3eoNIwd9j3gzg76ACZ18pDqa"
model_path = "model_multiclass.h5"

if not os.path.exists(model_path):
    st.info("⏳ Mendownload model dari Google Drive...")
    gdown.download(model_url, model_path, quiet=False)
    st.success("✅ Model berhasil didownload!")

# --- Step 2: Load Model ---
model = load_model(model_path)
labels = ['Normal', 'Katarak', 'Glaukoma', 'Diabetes']

# --- Step 3: Streamlit UI ---
st.title("🧠 Deteksi Penyakit Mata dari Citra Retina")
st.markdown("Upload gambar retina untuk klasifikasi otomatis: **Normal**, **Katarak**, **Glaukoma**, atau **Diabetes Retina**.")

uploaded_file = st.file_uploader("📤 Upload Gambar Retina", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar Diunggah", use_column_width=True)

    img = img.resize((150, 150))  # Sesuaikan dengan input model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # prediction = model.predict(img_array)[0]
    # predicted_label = labels[np.argmax(prediction)]
    # confidence = np.max(prediction)

    # st.subheader("🔍 Hasil Prediksi")
    # st.write(f"**Kelas:** {predicted_label}")
    # st.write(f"**Confidence:** {confidence:.2f}")

    # st.markdown("### 📊 Semua Probabilitas:")
    # for i in range(len(labels)):
    #     st.write(f"{labels[i]}: {prediction[i]:.2f}")
# --- HANYA JIKA MODEL ANDA ADALAH KLASIFIKASI BINER ---

# Definisikan ulang label Anda menjadi 2 kelas
binary_labels = ['Normal', 'Katarak'] 

# Model akan menghasilkan satu angka (misal, mendekati 0 untuk Normal, mendekati 1 untuk Katarak)
prediction_value = model.predict(img_array)[0][0] # Ambil satu-satunya nilai dari output

# Gunakan ambang batas (threshold) 0.5 untuk menentukan kelas
if prediction_value < 0.5:
    predicted_label = binary_labels[1] # Normal
    confidence = 1 - prediction_value
else:
    predicted_label = binary_labels[0] # Katarak
    confidence = prediction_value

st.subheader("🔍 Hasil Prediksi")
st.write(f"**Kelas:** {predicted_label}")
st.write(f"**Confidence:** {confidence:.2%}") # Tampilkan sebagai persentase

st.markdown("### 📊 Skor Prediksi:")
st.write(f"Nilai output model (0 = {binary_labels[0]}, 1 = {binary_labels[1]}): {prediction_value:.4f}")
