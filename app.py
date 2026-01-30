import streamlit as st
import pickle

st.title("Analisis Sentimen Berita Ekonomi")

try:
    nb = pickle.load(open("model_nb.pkl", "rb"))
    svm = pickle.load(open("model_svm.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    st.success("Model berhasil dimuat")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()
