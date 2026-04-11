import streamlit as st
import numpy as np
import re
import pickle
import joblib

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from textblob import TextBlob
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


# ── Ön İşleme & Tokenizasyon ────────────────────────────────────
def ekkok(text):
    words = TextBlob(text).words
    return [word.lemmatize() for word in words if word.lower() not in stop_words]


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.replace('\n', '').replace('\r', '')
    return text


# ── Model & Vectorizer Yükleme ───────────────────────────────────
@st.cache_resource
def load_model():
    from tensorflow.keras.models import load_model as keras_load_model
    try:
        return keras_load_model("sentiment_model.h5")
    except FileNotFoundError:
        try:
            import joblib
            return joblib.load("sentiment_model.joblib")
        except FileNotFoundError:
            return None


@st.cache_resource
def load_vectorizer():
    try:
        with open("vectorizer.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        try:
            return joblib.load("vectorizer.joblib")
        except FileNotFoundError:
            return None


# ── Tahmin Fonksiyonu ────────────────────────────────────────────
def predict_sentiment(text, model, vectorizer):
    cleaned = preprocess_text(text)
    features = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(features, verbose=0)
    confidence = float(prediction[0][0])
    is_positive = confidence > 0.5
    pos_score = confidence if is_positive else 1 - confidence
    neg_score = 1 - confidence if is_positive else confidence
    return is_positive, pos_score, neg_score


# ── Sayfa Ayarları ───────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 IMDB Sentiment Analysis",
    page_icon="🎥",
    layout="centered",
)

st.title("🎬 IMDB Review Sentiment Analysis")
st.write("Bu uygulama, IMDB film yorumlarının duygusunu yapay zekâ ile analiz eder.")

# ── Örnek Yorumlar ───────────────────────────────────────────────
with st.expander("📑 Örnek Yorumlar ile Test Edin"):
    st.markdown("**🟢 Olumlu Yorum Örneği:**")
    st.code(
        "After the success of Die Hard and its sequels it's no surprise really that "
        "Cliffhanger delivered as plenty of thrills. You've got to love John Lithgow's "
        "sneery evilness — best of all the permanently harassed and hapless turncoat agent.",
        language=None,
    )
    st.markdown("**🔴 Olumsuz Yorum Örneği:**")
    st.code(
        "I had the terrible misfortune of having to view this movie in its entirety. "
        "All I have to say is save your time and money. This has got to be the worst "
        "movie of all time. The story is not interesting at all.",
        language=None,
    )

# ── Kullanıcı Girişi ─────────────────────────────────────────────
st.subheader("✍️ Film Yorumunuzu Girin")
user_review = st.text_area(
    label="Analiz edilecek yorum:",
    height=200,
    placeholder="Film yorumunu buraya yapıştırın...",
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([2, 3, 2])
with col2:
    analyze_btn = st.button("🔍 ANALİZ ET", use_container_width=True, type="primary")

st.markdown("---")

# ── Sonuç ────────────────────────────────────────────────────────
if analyze_btn:
    if not user_review or user_review.strip() == "":
        st.warning("⚠️ Lütfen analiz etmek için bir yorum girin.")
    else:
        with st.spinner("🔄 Yorum analiz ediliyor..."):
            model = load_model()
            vectorizer = load_vectorizer()

            if model is None:
                st.error("❌ Model dosyası bulunamadı. `sentiment_model.h5` veya `sentiment_model.joblib` dosyasının mevcut olduğundan emin olun.")
            elif vectorizer is None:
                st.error("❌ Vectorizer dosyası bulunamadı. `vectorizer.pkl` veya `vectorizer.joblib` dosyasının mevcut olduğundan emin olun.")
            else:
                try:
                    is_positive, pos_score, neg_score = predict_sentiment(user_review, model, vectorizer)

                    if is_positive:
                        st.success(f"✅ Sonuç: **OLUMLU** yorum — Güven: %{pos_score * 100:.1f}")
                    else:
                        st.error(f"🚨 Sonuç: **OLUMSUZ** yorum — Güven: %{neg_score * 100:.1f}")

                    st.markdown("#### 📊 Güven Dağılımı")
                    col_p, col_n = st.columns(2)
                    with col_p:
                        st.metric(label="✅ Olumlu", value=f"%{pos_score * 100:.1f}")
                        st.progress(pos_score)
                    with col_n:
                        st.metric(label="❌ Olumsuz", value=f"%{neg_score * 100:.1f}")
                        st.progress(neg_score)

                except Exception as e:
                    st.error(f"❌ Analiz sırasında hata oluştu: {e}")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.85rem;'>"
    "🎬 IMDB Sentiment Analysis &copy; 2026 | Developed with ❤️ by tugcesi"
    "</div>",
    unsafe_allow_html=True,
)
