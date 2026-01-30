import streamlit as st
import pickle
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


nb = pickle.load(open("model_nb.pkl", "rb"))
svm = pickle.load(open("model_svm.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
