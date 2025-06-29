import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re

with open('animcond_model.sav', 'rb') as f:
    model = pickle.load(f)

with open('animcond_vectorizer.sav', 'rb') as f:
    vectorizer = pickle.load(f)

with open('animcond_labelenc.sav', 'rb') as f:
    label_encoder = pickle.load(f)

df = pd.read_csv('data.csv')

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['combined_symptoms'] = df[['symptoms1', 'symptoms2', 'symptoms3', 'symptoms4', 'symptoms5']].astype(str).agg(' '.join, axis=1)
df['processed_symptoms'] = df['combined_symptoms'].apply(preprocess)

st.title("🐾 Periksa Kondisi Hewanmu")
st.markdown("Masukkan jenis hewan dan gejala untuk memprediksi apakah penyakit tersebut berbahaya")

animal = st.text_input("Jenis Hewan (eng):")
symptoms = st.text_area("Masukkan gejala (eng):")

if st.button("Prediksi"):
    if not animal or not symptoms:
        st.warning("Lengkapi jenis hewan dan gejala terlebih dahulu.")
    else:
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
                st.warning("❗ Prediksi tidak dapat ditentukan dengan pasti. Deskripsikan gejala lebih jelas.")
            else:
                best_row = filtered_df.iloc[best_idx]
                st.subheader("🔍 Hasil Prediksi")
                st.write(f"**Hewan:** {animal}")
                st.write(f"**Gejala Paling Mirip:** {best_row['combined_symptoms']}")
                st.success(f"**Status Berbahaya:** {best_row['Dangerous']}")

                # ⬇️ Tambahkan pengecekan danger_status di sini
                raw_danger = best_row['Dangerous']
                if pd.isna(raw_danger):
                    danger_status = 'unknown'
                else:
                    danger_status = str(raw_danger).strip().lower()

                if danger_status == 'yes':
                    st.error("🚨 Gejala ini tergolong **berbahaya**. Segera periksakan hewan ke dokter hewan!")
                elif danger_status == 'no':
                    st.info("✅ Gejala tidak tergolong berbahaya. Tetap pantau kondisi hewan secara berkala.")
                else:
                    st.warning("⚠️ Status bahaya tidak diketahui. Silakan konsultasikan dengan dokter hewan.")
