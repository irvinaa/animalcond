import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- Load models ---
with open('animcond_model.sav', 'rb') as f:
    model = pickle.load(f)

with open('animcond_vectorizer.sav', 'rb') as f:
    vectorizer = pickle.load(f)

with open('animcond_labelenc.sav', 'rb') as f:
    label_encoder = pickle.load(f)

# --- Load DataFrame (untuk referensi input & similarity) ---
df = pd.read_csv('data.csv')

# --- Preprocessing Function ---
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Gabungkan kolom gejala (pastikan sama seperti di Colab)
df['combined_symptoms'] = df[['symptoms1', 'symptoms2', 'symptoms3', 'symptoms4', 'symptoms5']].astype(str).agg(' '.join, axis=1)
df['processed_symptoms'] = df['combined_symptoms'].apply(preprocess)

# --- UI ---
st.title("🐾 Sistem Klasifikasi Bahaya Penyakit Hewan")
st.markdown("Masukkan jenis hewan dan gejala untuk memprediksi apakah penyakit tersebut berbahaya.")

animal = st.text_input("Jenis Hewan (dalam bahasa Inggris):")
symptoms = st.text_area("Masukkan gejala (dalam bahasa Inggris):")

if st.button("Prediksi"):
    if not animal or not symptoms:
        st.warning("Silakan lengkapi jenis hewan dan gejala terlebih dahulu.")
    else:
        processed_input = preprocess(symptoms)
        input_vector = vectorizer.transform([processed_input])

        # Filter data hewan
        filtered_df = df[df['AnimalName'].str.lower() == animal.lower()]

        if filtered_df.empty:
            st.error(f"❌ Data untuk hewan '{animal}' tidak ditemukan.")
        else:
            filtered_vectors = vectorizer.transform(filtered_df['processed_symptoms'])
            similarities = cosine_similarity(input_vector, filtered_vectors).flatten()
            best_idx = similarities.argmax()
            best_similarity = similarities[best_idx]

            if best_similarity < 0.2:
                st.warning("❗ Prediksi tidak dapat ditentukan dengan pasti. Coba deskripsikan gejala lebih jelas.")
            else:
                best_row = filtered_df.iloc[best_idx]
                st.subheader("🔍 Hasil Prediksi")
                st.write(f"**Hewan:** {animal}")
                st.write(f"**Gejala Input:** {symptoms}")
                st.write(f"**Gejala Paling Mirip (dalam data):** {best_row['combined_symptoms']}")
                st.write(f"**Similarity Score:** {round(best_similarity, 3)}")
                st.success(f"**Status Berbahaya:** {best_row['Dangerous']}")
