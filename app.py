import streamlit as st
st.write("APP BERHASIL DIMUAT")
import pickle
nb = pickle.load(open("model_nb.pkl", "rb"))
svm = pickle.load(open("model_svm.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory





