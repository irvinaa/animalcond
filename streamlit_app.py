import pickle
import numpy as np
import streamlit as st

model = pickle.load(open('animcond_model.sav','rb'))

st.title('❗❗Lihat Bahaya Penyakit Hewanmu❗❗')
