# NewsLIME: Fast & Lightweight Fake News Detection

## Technical Report

### 1. Introduction

### 1. Introduction

NewsLIME is a **hybrid high-performance fake news detection system** that bridges the gap between speed and trust. It delivers **zero-latency classification** while retaining **full explainability** via LIME.

The system is designed on three pillars:
1.  **Explainability**: Users must understand *why* an article is flagged.
2.  **Performance**: Inference must be instant (~50ms).
3.  **Simplicity**: Deployment should be lightweight, with no heavy external dependencies like NLTK.

### 2. Dataset

**Source:** WELFake Dataset ([Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification))

The WELFake dataset is a merged corpus compiled from four widely used fake news sources: Kaggle, McIntire, Reuters, and BuzzFeed Political. It relies on ~72,000 news articles.

| Property | Value |
|----------|-------|
| Total samples | 72,134 |
| Real articles | 37,106 (51.4%) |
| Fake articles | 35,028 (48.6%) |
| Features | `title`, `text`, `label` |
| Format | CSV, ~234 MB |
| License | CC BY 4.0 |

The dataset is nearly balanced, which reduces the risk of class-biased predictions. Missing values are filled with empty strings during preprocessing.

### 3. Text Preprocessing Pipeline

To achieve both high speed and accuracy without external downloads, we optimized the pipeline:

1.  **Lowercasing** — Normalize case.
2.  **Regex Cleaning** — Strip HTML and non-alphabetic characters.
3.  **Vectorization-based Stopword Removal** — Instead of downloading the 10MB+ NLTK corpus at runtime, we leverage Scikit-Learn's built-in `stop_words='english'` list within the TF-IDF vectorizer.

This approach eliminates the "missing corpus" errors common in cloud deployments while maintaining the model's ability to ignore noise words.

### 4. Feature Extraction

**Method:** TF-IDF (Term Frequency-Inverse Document Frequency)

| Parameter | Value |
|-----------|-------|
| `max_features` | 5,000 |
| `ngram_range` | (1, 2) |
| `stop_words` | 'english' |

We extract unigrams and bigrams to capture key phrases (e.g., "breaking news", "official source"). Limiting the vocabulary to 5,000 features ensures the model file remains lightweight (~25MB compressed) and inference stays under 50ms.

### 5. Model Architecture

**Classifier:** Random Forest

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 200 |
| `random_state` | 42 |
| `n_jobs` | -1 (all cores) |

Random Forest provides excellent accuracy on tabular TF-IDF data and naturally handles non-linear relationships between words and labels. It is robust to overfitting and does not require GPU acceleration.

### 6. Training and Evaluation

**Data split:**

| Split | Proportion | Samples |
|-------|------------|---------|
| Training | 80% | 57,707 |
| Validation | 10% | 7,213 |
| Test | 10% | 7,214 |

**Performance Metrics:**

| Metric | Real | Fake |
|--------|------|------|
| Precision | 0.95 | 0.97 |
| Recall | 0.97 | 0.95 |
| F1-Score | 0.96 | 0.96 |

**Overall Accuracy:**
- **Training**: 100%
- **Validation**: 96.22%
- **Test**: 96.01%

The model shows strong generalization with consistent performance across validation and test sets.

### 7. System Architecture

```
User Input (article text)
        |
        v
    Simple Regex Cleaning      -- Preprocessing (No NLTK)
        |
        v
    TfidfVectorizer            -- Feature extraction (vectorizer.joblib)
        |
        v
    RandomForest               -- Classification (model.joblib)
        |
        v
    LIME Explainer             -- (On-Demand) Word Importance Analysis
        |
        v
    Result Display             -- Streamlit UI
```

### 8. Explainability (LIME)

Despite the lightweight architecture, NewsLIME retains **Local Interpretable Model-agnostic Explanations (LIME)**.

-   **Why it matters**: A credibility score alone is a "black box." LIME perturbs the input text (removing words randomly) and queries the model to see which words flip the prediction.
-   **Implementation**: By using the `LimeTextExplainer` on top of our optimized prediction pipeline, we generate explanations dynamically without needing heavy NLP preprocessing. The Explainer views the text as a simple bag of words, matching our vectorizer's approach.

### 9. Dashboard Features

The Streamlit interface (`app.py`) is designed for minimalism:

- **Clean UI**: Single-column layout with a clear input area.
- **Instant Analysis**: Results appear immediately upon clicking "Analyze".
- **Confidence Meter**: Visual bar showing the probability of the predicted class.
- **Zero Clutter**: No complex sidebars or configuration toggles.

### 9. Project Structure

```
NewsLIME/
├── app.py                  # Minimalist Streamlit dashboard
├── train_model.py          # Optimized training script
├── download_data.py        # Dataset download utility
├── model.joblib            # Compressed Random Forest model
├── vectorizer.joblib       # Compressed TF-IDF vectorizer
├── requirements.txt        # Minimal dependencies
├── REPORT.md               # This technical report
└── data/
    └── WELFake_Dataset.csv # Training data
```

### 10. Dependencies

The dependency list has been significantly reduced:

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >= 2.0.0 | Data handling |
| scikit-learn | >= 1.3.0 | ML pipeline |
| joblib | >= 1.3.0 | Model loading |
| streamlit | >= 1.30.0 | Web interface |
| numpy | >= 1.24.0 | Math operations |

**Removed**: `nltk`, `lime`, `matplotlib`, `scipy` (explicit dependency).

### 11. Reproduction

```bash
pip install -r requirements.txt
python download_data.py       # Download data
python train_model.py         # Train & Save (auto-compresses models)
streamlit run app.py          # Launch
```

### 12. Limitations

- **English Only**: The `stop_words='english'` setting restricts the model to English articles.
- **Static vocabulary**: Words not in the top 5,000 frequent terms are ignored.
- **No deep context**: Unlike Transformer models (BERT/GPT), this model relies on word frequency patterns rather than deep semantic understanding.
