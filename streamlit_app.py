import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Load model & data
with open('animcond_vectorizer.sav', 'rb') as f:
    vectorizer = pickle.load(f)

with open('animcond_model.sav', 'rb') as f:
    svm = pickle.load(f)

# Fungsi preprocessing sesuai notebook
def preprocess(text):
    # Ganti ini dengan versi lengkap preprocessing kamu
    return text.lower()

# UI
st.title("🔍 Sistem Klasifikasi Bahaya Penyakit Hewan")
st.write("Masukkan jenis hewan dan gejala yang dialami.")

animal = st.text_input("Jenis Hewan (dalam Bahasa Inggris)")
symptoms = st.text_area("Gejala yang dialami (dalam Bahasa Inggris)")

if st.button("Prediksi"):
    if animal and symptoms:
        processed_input = preprocess(symptoms)
        input_vector = vectorizer.transform([processed_input])
        
        filtered_df = df[df['AnimalName'].str.lower() == animal.lower()]

        if filtered_df.empty:
            st.error(f"❌ Data untuk hewan '{animal}' tidak ditemukan.")
        else:
            filtered_vectors = vectorizer.transform(filtered_df['processed_symptoms'])
            similarities = cosine_similarity(input_vector, filtered_vectors).flatten()

            best_idx = similarities.argmax()
            best_similarity = similarities[best_idx]

            if best_similarity < 0.2:
                st.warning("⚠️ Gejala tidak cukup mirip. Pastikan penulisan gejala benar.")
            else:
                best_row = filtered_df.iloc[best_idx]
                st.subheader("✅ Hasil Prediksi")
                st.write(f"**Hewan:** {animal}")
                st.write(f"**Gejala Input:** {symptoms}")
                st.write(f"**Gejala Paling Mirip:** {best_row['combined_symptoms']}")
                st.write(f"**Similarity Score:** {round(best_similarity, 3)}")
                st.write(f"**Klasifikasi Berbahaya:** {best_row['Dangerous']}")
    else:
        st.warning("❗ Mohon isi kedua kolom input.")
