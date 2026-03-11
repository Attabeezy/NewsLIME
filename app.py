"""NewsLIME: Explainable Fake News Detector — Streamlit Dashboard."""

import os
import random
import re

import joblib
import matplotlib
import matplotlib.pyplot as plt
import requests
import streamlit as st
import streamlit.components.v1 as components
from lime.lime_text import LimeTextExplainer

BASE_DIR = os.path.dirname(__file__)


@st.cache_resource
def load_model():
    """Load the trained model and vectorizer from disk."""
    model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.joblib"))
    return model, vectorizer


def clean_text(text):
    """Simple text cleaning: lowercase and strip HTML/special chars."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text


def predict_proba(texts):
    """Prediction function compatible with LIME (takes list of raw strings)."""
    model, vectorizer = load_model()
    # Cleaning is minimal now as vectorizer handles stopwords
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


def fetch_random_article(api_key: str) -> dict | None:
    """Fetch a random top-headline article from NewsAPI. Returns dict with title and body."""
    url = "https://newsapi.org/v2/top-headlines"
    params = {"country": "us", "pageSize": 20, "apiKey": api_key}
    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        articles = [
            a
            for a in resp.json().get("articles", [])
            if a.get("content") or a.get("description")
        ]
        if not articles:
            return None
        article = random.choice(articles)
        parts = [
            article.get("title", ""),
            article.get("description", ""),
            article.get("content", ""),
        ]
        body = "\n\n".join(p for p in parts if p)
        # Strip the "[+N chars]" truncation marker NewsAPI appends
        body = re.sub(r"\s*\[\+\d+ chars\]$", "", body)
        return {
            "title": article.get("title", ""),
            "body": body,
            "source": article.get("source", {}).get("name", ""),
        }
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 401:
            raise ValueError("Invalid NewsAPI key.") from e
        raise


def build_lime_css(is_dark: bool) -> str:
    """Return CSS to inject into the LIME HTML iframe for readable styling."""
    if is_dark:
        lime_bg, lime_text, border_col = "#1a1a2e", "#e8e8f0", "#3a3a5c"
    else:
        lime_bg, lime_text, border_col = "#ffffff", "#1a1a1a", "#d0d0d8"
    return f"""<style>
    html, body {{
        background-color: {lime_bg} !important;
        color: {lime_text} !important;
        font-family: 'Source Sans Pro', sans-serif;
        margin: 0; padding: 8px;
    }}
    table, thead, tbody, tr, td, th {{
        color: {lime_text} !important;
        border-color: {border_col} !important;
    }}
    svg text {{ fill: {lime_text} !important; }}
    .lime.top_div {{ background-color: {lime_bg} !important; color: {lime_text} !important; }}
    p {{ color: {lime_text} !important; }}
    .lime span {{ color: {lime_text} !important; }}
</style>"""


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
            }}
            .stTextArea textarea {{
                background-color: {secondary_bg} !important;
                color: {text_color} !important;
                border: 1px solid {text_color}33;
            }}
            .stMarkdown p,
            .stMarkdown li,
            .stMarkdown h1,
            .stMarkdown h2,
            .stMarkdown h3,
            .stMarkdown h4,
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] li {{
                color: {text_color} !important;
            }}
            [data-testid="stMetricLabel"],
            [data-testid="stMetricLabel"] p,
            [data-testid="stMetricValue"],
            [data-testid="stMetricDelta"] {{
                color: {text_color} !important;
            }}
            [data-testid="stCaptionContainer"] p,
            .stCaption {{
                color: {text_color} !important;
                opacity: 0.65;
            }}
            [data-testid="stHeading"] h2,
            [data-testid="stHeading"] h3 {{
                color: {text_color} !important;
            }}
            [data-testid="stVerticalBlockBorderWrapper"] {{
                background-color: {secondary_bg} !important;
                border-color: {text_color}22 !important;
            }}
            [data-testid="stTabs"] button[role="tab"] {{
                color: {text_color} !important;
            }}
            hr {{
                border-color: {text_color}22 !important;
            }}
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] li,
            [data-testid="stSidebar"] label {{
                color: {text_color} !important;
                font-size: 0.93rem;
                line-height: 1.55;
            }}
            [data-testid="stMain"] {{
                background-color: {bg} !important;
            }}
            .main .block-container {{
                background-color: {bg} !important;
            }}
            [data-testid="stBaseButton-primary"] {{
                background-color: #4f8ef7 !important;
                color: #ffffff !important;
                border-color: #4f8ef7 !important;
            }}
            [data-testid="stBaseButton-secondary"] {{
                background-color: {secondary_bg} !important;
                color: {text_color} !important;
                border-color: {text_color}33 !important;
            }}
            [data-testid="stProgressBar"] > div {{
                background-color: {secondary_bg} !important;
            }}
            [data-testid="stProgressBar"] > div > div {{
                background-color: #4f8ef7 !important;
            }}
            [data-testid="stTabsContent"] {{
                background-color: {bg} !important;
            }}
            [data-testid="stToggle"] label {{
                color: {text_color} !important;
            }}
            [data-testid="stTextInput"] input {{
                background-color: {secondary_bg} !important;
                color: {text_color} !important;
                border-color: {text_color}33 !important;
            }}
            [data-testid="stTextInput"] label {{
                color: {text_color} !important;
            }}
            [data-testid="stSpinner"] p {{
                color: {text_color} !important;
            }}
            [data-testid="stAlert"] {{
                background-color: {secondary_bg} !important;
            }}
            [data-testid="stAlert"] p {{
                color: {text_color} !important;
            }}
            [data-testid="stHeading"] h1,
            [data-testid="stHeading"] h4 {{
                color: {text_color} !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Explainable Fake News Detector

        This tool uses a Machine Learning model (**TF-IDF + Random Forest**)
        to classify news articles as **Real** or **Fake**.

        It provides word-level explanations using **LIME** (Local Interpretable Model-agnostic Explanations).

        ---

        #### How to read the explanation

        - Words highlighted in **red** push the prediction toward **Fake**
        - Words highlighted in **green** push the prediction toward **Real**
        - The bar lengths show how strongly each word influences the result

        ---

        #### About the model

        - Trained on ~72,000 news articles (**WELFake dataset**)
        - Accuracy: **96%** on held-out test set
        - Uses **TF-IDF vectorization** with bigrams
        - **Random Forest** classifier (200 estimators)
        """
    )

    st.markdown("---")
    st.markdown("#### Random Article")
    news_api_key = st.text_input(
        "NewsAPI Key",
        type="password",
        value=os.getenv("NEWS_API_KEY", ""),
        placeholder="Paste your free key from newsapi.org",
        help="Get a free key at newsapi.org. Used only to fetch random articles.",
    )

    st.divider()
    if st.button("Clear Input", use_container_width=True):
        st.session_state["news_input"] = ""
        st.rerun()

# ── Main Area ────────────────────────────────────────────────────────────────

header_bg = (
    "linear-gradient(90deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%)"
    if theme
    else "linear-gradient(90deg, #e8edf5 0%, #dce4f0 60%, #c8d6ec 100%)"
)

st.markdown(
    f"""
    <div style="
        background: {header_bg};
        border-left: 4px solid #4f8ef7;
        border-radius: 6px;
        padding: 20px 24px 16px 24px;
        margin-bottom: 16px;
    ">
        <h1 style="color: {text_color}; margin: 0 0 6px 0; font-size: 2rem; font-weight: 700; letter-spacing: -0.5px;">NewsLIME</h1>
        <p style="color: {text_color}; opacity: 0.7; margin: 0; font-size: 1rem;">
            Explainable Fake News Detection — paste an article to analyze its credibility.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    news_text = st.text_area(
        "News Article",
        height=220,
        key="news_input",
        placeholder="Paste the full news article text here...",
        label_visibility="collapsed",
    )
    col_rand, col_btn, col_hint = st.columns([1, 1, 2])
    with col_rand:
        fetch_btn = st.button(
            "Random Article",
            use_container_width=True,
            help="Pull a random headline via NewsAPI",
        )
    with col_btn:
        analyze = st.button("Analyze Article", type="primary", use_container_width=True)
    with col_hint:
        st.caption(
            "Model: TF-IDF + Random Forest (96% accuracy on WELFake). LIME explanation takes ~5 seconds."
        )

if fetch_btn:
    if not news_api_key:
        st.warning("Enter a NewsAPI key in the sidebar to fetch random articles.")
    else:
        with st.spinner("Fetching random article..."):
            try:
                article = fetch_random_article(news_api_key)
                if article:
                    st.session_state["news_input"] = article["body"]
                    st.toast(
                        f"Loaded: {article['title'][:80]}{'…' if len(article['title']) > 80 else ''}",
                        icon="📰",
                    )
                    st.rerun()
                else:
                    st.warning("No articles returned. Try again.")
            except ValueError as e:
                st.error(str(e))
            except Exception:
                st.error(
                    "Failed to fetch article. Check your API key and internet connection."
                )

if analyze and news_text.strip():
    model, vectorizer = load_model()

    with st.spinner("Analyzing article..."):
        cleaned = clean_text(news_text)
        tfidf = vectorizer.transform([cleaned])
        proba = model.predict_proba(tfidf)[0]
        real_prob, fake_prob = proba[0], proba[1]
        credibility = real_prob * 100
        level, color = risk_level(fake_prob)
        pred_label = "REAL" if real_prob > fake_prob else "FAKE"

    # ── Results ──────────────────────────────────────────────────────────
    st.divider()

    with st.container(border=True):
        st.markdown(
            f"<p style='font-size:0.8rem; text-transform:uppercase; letter-spacing:1px; "
            f"color:{text_color}; opacity:0.55; margin:0 0 12px 0;'>Analysis Results</p>",
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Credibility Score", f"{credibility:.1f}%")
            st.progress(credibility / 100)
            st.caption(f"Real probability: {real_prob:.1%}")

        with col2:
            badge_bg = "#1b5e20" if pred_label == "REAL" else "#b71c1c"
            badge_text_color = "#a5d6a7" if pred_label == "REAL" else "#ef9a9a"
            st.markdown(
                f"""
                <div style="margin-top: 4px;">
                    <p style="font-size:0.85rem; color:{text_color}; opacity:0.65; margin:0 0 6px 0;">Prediction</p>
                    <span style="
                        background-color: {badge_bg};
                        color: {badge_text_color};
                        font-size: 1.4rem;
                        font-weight: 700;
                        padding: 6px 18px;
                        border-radius: 4px;
                        letter-spacing: 1px;
                        display: inline-block;
                    ">{pred_label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"Fake probability: {fake_prob:.1%}")

        with col3:
            risk_colors = {
                "High Risk": "#f44336",
                "Medium Risk": "#ff9800",
                "Low Risk": "#4caf50",
            }
            st.metric("Risk Level", level)
            st.markdown(
                f"<p style='color:{risk_colors[level]}; font-weight:600; font-size:0.9rem; margin-top:2px;'>{level}</p>",
                unsafe_allow_html=True,
            )

    # ── LIME Explanation ─────────────────────────────────────────────────
    st.divider()

    with st.container(border=True):
        st.subheader("Why this prediction?")
        st.caption(
            "Red highlights push toward Fake. Green highlights push toward Real. "
            "Bar lengths show word influence strength."
        )

        with st.spinner("Generating LIME explanation (~5s)..."):
            explanation = get_lime_explanation(news_text)

        tab_text, tab_chart = st.tabs(["Text Highlight", "Feature Importance"])

        with tab_text:
            html = explanation.as_html()
            html = html.replace("<head>", "<head>" + build_lime_css(is_dark=theme), 1)
            components.html(html, height=520, scrolling=True)

        with tab_chart:
            facecolor = "#1a1a2e" if theme else "#ffffff"
            textcolor = "#e8e8f0" if theme else "#1a1a1a"
            for key in ["figure.facecolor", "axes.facecolor"]:
                matplotlib.rcParams[key] = facecolor
            for key in ["text.color", "axes.labelcolor", "xtick.color", "ytick.color"]:
                matplotlib.rcParams[key] = textcolor
            fig = explanation.as_pyplot_figure(label=1)
            fig.set_size_inches(8, 4)
            fig.suptitle("Feature importance for 'Fake' class", fontsize=11, y=1.01)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption(
                "Positive bars push toward Real. Negative bars push toward Fake."
            )

elif analyze:
    st.warning("Please paste a news article before clicking Analyze.")
