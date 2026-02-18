"""NewsLIME: Explainable Fake News Detector — Streamlit Dashboard."""

import os
import re

import joblib
import numpy as np
import streamlit as st
from lime.lime_text import LimeTextExplainer
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))
BASE_DIR = os.path.dirname(__file__)


@st.cache_resource
def load_model():
    """Load the trained model and vectorizer from disk."""
    model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.joblib"))
    return model, vectorizer


def clean_text(text):
    """Clean text: lowercase, strip HTML, remove special chars and stopwords."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


def predict_proba(texts):
    """Prediction function compatible with LIME (takes list of raw strings)."""
    model, vectorizer = load_model()
    cleaned = [clean_text(t) for t in texts]
    tfidf = vectorizer.transform(cleaned)
    return model.predict_proba(tfidf)


def get_lime_explanation(text, num_features=10):
    """Generate a LIME explanation for the given text."""
    explainer = LimeTextExplainer(class_names=["Real", "Fake"])
    explanation = explainer.explain_instance(
        text, predict_proba, num_features=num_features, num_samples=500
    )
    return explanation


def risk_level(fake_prob):
    """Return risk level label and color based on fake probability."""
    if fake_prob >= 0.75:
        return "High Risk", "red"
    elif fake_prob >= 0.4:
        return "Medium Risk", "orange"
    else:
        return "Low Risk", "green"


# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="NewsLIME — Fake News Detector", layout="wide")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("NewsLIME")

    theme = st.toggle("Dark Mode", value=True)
    if theme:
        bg, text_color, secondary_bg = "#0e1117", "#fafafa", "#262730"
    else:
        bg, text_color, secondary_bg = "#ffffff", "#1a1a1a", "#f0f2f6"

    st.markdown(
        f"""
        <style>
            .stApp, [data-testid="stAppViewContainer"] {{
                background-color: {bg};
                color: {text_color};
            }}
            [data-testid="stSidebar"] {{
                background-color: {secondary_bg};
                color: {text_color};
            }}
            .stTextArea textarea {{
                background-color: {secondary_bg};
                color: {text_color};
            }}
            .stMarkdown, .stCaption, p, span, label, h1, h2, h3 {{
                color: {text_color} !important;
            }}
            [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
                color: {text_color} !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        **Explainable Fake News Detector**

        This tool uses a Machine Learning model (TF-IDF + Random Forest)
        to classify news articles as **Real** or **Fake**, and provides
        word-level explanations using **LIME** (Local Interpretable
        Model-agnostic Explanations).

        ---

        ### How to read the explanation

        - Words highlighted in **:red[red]** push the prediction toward **Fake**
        - Words highlighted in **:green[green]** push the prediction toward **Real**
        - The bar lengths show how strongly each word influences the result

        ---

        ### About the model

        - Trained on ~72,000 news articles (WELFake dataset)
        - Accuracy: **96%** on held-out test set
        - Uses TF-IDF vectorization with bigrams
        - Random Forest classifier (200 estimators)
        """
    )

    if st.button("Clear Input", use_container_width=True):
        st.session_state["news_input"] = ""
        st.rerun()

# ── Main Area ────────────────────────────────────────────────────────────────

st.header("Fake News Detector")
st.markdown("Paste a news article below and click **Analyze** to check its credibility.")

news_text = st.text_area(
    "News Article",
    height=250,
    key="news_input",
    placeholder="Paste the full news article text here...",
)

analyze = st.button("Analyze", type="primary", use_container_width=True)

if analyze and news_text.strip():
    model, vectorizer = load_model()

    with st.spinner("Analyzing article..."):
        # Predict
        cleaned = clean_text(news_text)
        tfidf = vectorizer.transform([cleaned])
        proba = model.predict_proba(tfidf)[0]
        real_prob, fake_prob = proba[0], proba[1]
        credibility = real_prob * 100

        # Risk level
        level, color = risk_level(fake_prob)

    # ── Results ──────────────────────────────────────────────────────────
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Credibility Score", f"{credibility:.1f}%")
    with col2:
        st.metric("Prediction", "Real" if real_prob > fake_prob else "Fake")
    with col3:
        st.markdown(f"### :{color}[{level}]")

    # ── LIME Explanation ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Explanation — Why this prediction?")
    st.caption(
        "The highlights below show which words most influenced the model's decision. "
        "**Red** words push toward Fake; **green** words push toward Real."
    )

    with st.spinner("Generating LIME explanation..."):
        explanation = get_lime_explanation(news_text)
        html = explanation.as_html()

    st.components.v1.html(html, height=800, scrolling=True)

elif analyze:
    st.warning("Please paste a news article before clicking Analyze.")
