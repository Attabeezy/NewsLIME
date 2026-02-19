"""Train a Fake News classifier using TF-IDF + Random Forest."""

import os
import re

import joblib
import nltk
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Download NLTK stopwords
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")


def clean_text(text):
    """Clean a text string: lowercase, strip HTML, remove special chars and stopwords."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)  # remove HTML tags
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters and spaces
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


def load_data():
    """Load WELFake_Dataset.csv and prepare DataFrame with labels."""
    df = pd.read_csv(os.path.join(DATA_DIR, "WELFake_Dataset.csv"))

    # Map: label 0=Fake -> 1, label 1=Real -> 0  (internal: 1=Fake, 0=Real)
    df["label"] = 1 - df["label"]

    # Handle missing values
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    # Combine title and text into a single content field
    df["content"] = df["title"] + " " + df["text"]

    return df


def train():
    """Train the TF-IDF + Random Forest pipeline and save artifacts."""
    df = load_data()
    print(f"Dataset: {len(df)} samples ({(df['label'] == 1).sum()} fake, {(df['label'] == 0).sum()} real)")


    df["content_clean"] = df["content"].apply(clean_text)
    print(f"{len(df)} samples cleaned")

    X = df["content_clean"]
    y = df["label"]

    X_train, X_, y_train, y_ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_, y_, test_size=0.5, random_state=42, stratify=y_
    )

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_valid_tfidf = vectorizer.transform(X_valid)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_tfidf, y_train)

    y_train_pred = model.predict(X_train_tfidf)
    y_valid_pred = model.predict(X_valid_tfidf)
    y_test_pred = model.predict(X_test_tfidf)

    print("\n=== Classification Report ===")
    print("Training Set:")
    print(classification_report(y_train, y_train_pred, target_names=["Real", "Fake"]))
    print("Validation Set:")
    print(classification_report(y_valid, y_valid_pred, target_names=["Real", "Fake"]))
    print("Test Set:")
    print(classification_report(y_test, y_test_pred, target_names=["Real", "Fake"]))

    print("=== Confusion Matrix ===")
    print("Training Set:")
    cm = confusion_matrix(y_train, y_train_pred)
    print(cm)
    print("Validation Set:")
    cm = confusion_matrix(y_valid, y_valid_pred)
    print(cm)
    print("Test Set:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)

    accuracy_train = np.mean(y_train_pred == y_train)
    accuracy_valid = np.mean(y_valid_pred == y_valid)
    accuracy_test = np.mean(y_test_pred == y_test)
    print(f"\nAccuracy: {accuracy_train:.4f} (Train), {accuracy_valid:.4f} (Validation), {accuracy_test:.4f} (Test)")

    if accuracy_test < 0.85:
        print("WARNING: Accuracy below 85%. Consider adjusting ngram_range or max_features.")

    # Save model and vectorizer
    model_path = os.path.join(BASE_DIR, "model.joblib")
    vec_path = os.path.join(BASE_DIR, "vectorizer.joblib")
    joblib.dump(model, model_path, compress=3)
    joblib.dump(vectorizer, vec_path, compress=3)
    print(f"\nModel saved to {model_path}")
    print(f"Vectorizer saved to {vec_path}")


if __name__ == "__main__":
    train()
