import streamlit as st
import pickle
import re

st.set_page_config(page_title="Analisis Sentimen Berita Ekonomi", layout="centered")

st.title("📊 Analisis Sentimen Berita Ekonomi CNBC Indonesia")
st.write("Masukkan judul berita ekonomi untuk memprediksi sentimen menggunakan Naïve Bayes dan SVM.")

# ================= LOAD MODEL =================
try:
    nb = pickle.load(open("model_nb.pkl", "rb"))
    svm = pickle.load(open("model_svm.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    st.success("Model berhasil dimuat")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ================= NLP TOOLS =================
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    stemmer = StemmerFactory().create_stemmer()
    stopword = StopWordRemoverFactory().create_stop_word_remover()
except Exception as e:
    st.error(f"Gagal memuat Sastrawi: {e}")
    st.stop()

# ================= PREPROCESS =================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", "", text)
    text = stopword.remove(text)
    text = stemmer.stem(text)
    return text

# ================= UI =================
text_input = st.text_area("📝 Masukkan Judul Berita Ekonomi")

if st.button("🔍 Prediksi Sentimen"):
    if text_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        clean_text = preprocess_text(text_input)
        vector = tfidf.transform([clean_text])

        pred_nb = nb.predict(vector)[0]
        pred_svm = svm.predict(vector)[0]

        st.subheader("📌 Hasil Prediksi")
        st.write("🟢 **Naïve Bayes** :", pred_nb)
        st.write("🔵 **SVM** :", pred_svm)
