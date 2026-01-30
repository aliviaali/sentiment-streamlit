import streamlit as st
import pickle
import re

st.title("Analisis Sentimen Berita Ekonomi CNBC Indonesia")
st.write("Aplikasi berhasil dimuat")

# ===== Load Model (SAFE) =====
try:
    nb = pickle.load(open("model_nb.pkl", "rb"))
    svm = pickle.load(open("model_svm.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    st.success("Model berhasil dimuat")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ===== Load NLP Tools =====
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    stemmer = StemmerFactory().create_stemmer()
    stopword = StopWordRemoverFactory().create_stop_word_remover()
except Exception as e:
    st.error(f"Gagal memuat Sastrawi: {e}")
    st.stop()

# ===== Preprocessing =====
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", "", text)
    text = stopword.remove(text)
    text = stemmer.stem(text)
    return text

# ===== UI =====
text_input = st.text_area("Masukkan judul berita ekonomi:")

if text_input:
    clean_text = preprocess_text(text_input)
    vector = tfidf.transform([clean_text])

    pred_nb = nb.predict(vector)[0]
    pred_svm = svm.predict(vector)[0]

    st.subheader("Hasil Prediksi Sentimen")
    st.write("Naïve Bayes :", pred_nb)
    st.write("SVM         :", pred_svm)
