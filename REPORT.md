# NewsLIME: Explainable Fake News Detection System

## Technical Report

### 1. Introduction

NewsLIME is a fake news detection system that combines a machine learning classifier with LIME (Local Interpretable Model-agnostic Explanations) to provide transparent, word-level reasoning behind each prediction. The system is deployed as an interactive Streamlit dashboard where users can paste news articles and receive both a classification result and a visual explanation of which words influenced the decision.

The core motivation is bridging the gap between accurate classification and user trust. A prediction alone is insufficient—users need to understand *why* a model considers an article fake or real.

### 2. Dataset

**Source:** WELFake Dataset ([Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification))

The WELFake dataset is a merged corpus compiled from four widely used fake news sources: Kaggle, McIntire, Reuters, and BuzzFeed Political. It was published in *IEEE Transactions on Computational Social Systems* (DOI: 10.1109/TCSS.2021.3068519).

| Property | Value |
|----------|-------|
| Total samples | 72,134 |
| Real articles | 37,106 (51.4%) |
| Fake articles | 35,028 (48.6%) |
| Features | `title`, `text`, `label` |
| Format | CSV, ~234 MB |
| License | CC BY 4.0 |

The dataset is nearly balanced, which reduces the risk of class-biased predictions. Missing values are minimal (558 null titles, 39 null texts) and are filled with empty strings during preprocessing.

### 3. Text Preprocessing Pipeline

All text undergoes the following cleaning steps before vectorization:

1. **Lowercasing** — Normalize case to reduce vocabulary size.
2. **HTML tag removal** — Strip any embedded HTML markup using regex (`<.*?>`).
3. **Special character removal** — Retain only alphabetic characters and whitespace (`[^a-z\s]`).
4. **Stopword removal** — Remove English stopwords using the NLTK corpus.
5. **Field concatenation** — The `title` and `text` fields are concatenated into a single `content` string before cleaning.

The same `clean_text()` function is used in both training (`train_model.py`) and inference (`app.py`) to ensure consistency.

### 4. Feature Extraction

**Method:** TF-IDF (Term Frequency-Inverse Document Frequency)

| Parameter | Value |
|-----------|-------|
| `max_features` | 5,000 |
| `ngram_range` | (1, 2) |

Unigrams and bigrams are extracted to capture both individual word importance and two-word phrases (e.g., "breaking news", "anonymous sources"). The vocabulary is capped at 5,000 features to balance expressiveness with computational cost.

The fitted `TfidfVectorizer` is serialized to `vectorizer.joblib` for use at inference time.

### 5. Model Architecture

**Classifier:** Random Forest

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 200 |
| `random_state` | 42 |
| `n_jobs` | -1 (all cores) |

Random Forest was chosen for its strong performance on TF-IDF features, resistance to overfitting via ensemble averaging, and native support for `predict_proba`—which LIME requires for generating explanations.

### 6. Training and Evaluation

**Data split:**

| Split | Proportion | Samples |
|-------|------------|---------|
| Training | 80% | 57,707 |
| Validation | 10% | 7,213 |
| Test | 10% | 7,214 |

Stratified splitting is used to maintain class proportions across all sets.

**Test set results:**

| Metric | Real | Fake |
|--------|------|------|
| Precision | 0.95 | 0.97 |
| Recall | 0.97 | 0.95 |
| F1-Score | 0.96 | 0.96 |

| | Predicted Real | Predicted Fake |
|---|---|---|
| **Actual Real** | 3,602 | 109 |
| **Actual Fake** | 179 | 3,324 |

**Overall test accuracy: 96.0%**

The model achieves balanced performance across both classes, with no significant bias toward either real or fake predictions.

### 7. Explainability with LIME

LIME (Local Interpretable Model-agnostic Explanations) generates per-prediction explanations by perturbing the input text and observing how the model's output changes.

**Configuration:**

| Parameter | Value |
|-----------|-------|
| `num_features` | 10 |
| `num_samples` | 500 |
| `class_names` | ["Real", "Fake"] |

For each prediction, LIME:
1. Randomly removes words from the input to create 500 perturbed samples.
2. Gets the model's probability estimates for each perturbed sample.
3. Fits a local linear model to identify which words most influenced the prediction.
4. Outputs a ranked list of words with their contribution direction and magnitude.

In the dashboard, words pushing toward "Fake" are highlighted in red, and words pushing toward "Real" are highlighted in green.

### 8. System Architecture

```
User Input (article text)
        |
        v
  clean_text()         -- Preprocessing
        |
        v
  TfidfVectorizer      -- Feature extraction (vectorizer.joblib)
        |
        v
  RandomForest          -- Classification (model.joblib)
        |
        v
  predict_proba()       -- Probability output [P(Real), P(Fake)]
        |
        v
  LimeTextExplainer     -- Word-level explanation
        |
        v
  Streamlit Dashboard   -- Credibility score, prediction, risk level, LIME visualization
```

### 9. Dashboard Features

The Streamlit interface (`app.py`) provides:

- **Text input area** for pasting news articles.
- **Credibility Score** — Probability of the article being real, displayed as a percentage.
- **Prediction label** — "Real" or "Fake" based on the higher probability class.
- **Risk level** — Color-coded severity indicator:
  - High Risk (red): fake probability >= 75%
  - Medium Risk (orange): fake probability 40-74%
  - Low Risk (green): fake probability < 40%
- **LIME explanation** — Interactive HTML visualization showing the top 10 most influential words with bar charts and highlighted text.
- **Dark/Light mode** toggle.
- **Clear Input** button to reset the form.

Model and vectorizer are loaded once and cached via `@st.cache_resource`.

### 10. Project Structure

```
NewsLIME/
├── app.py                  # Streamlit dashboard
├── train_model.py          # Model training script
├── download_data.py        # Dataset download utility
├── model.joblib            # Trained Random Forest classifier
├── vectorizer.joblib       # Fitted TF-IDF vectorizer
├── requirements.txt        # Python dependencies
├── TECHNICAL_REPORT.md     # This report
└── data/
    └── WELFake_Dataset.csv # Training data (72,134 articles)
```

### 11. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >= 2.0.0 | Data loading and manipulation |
| scikit-learn | >= 1.3.0 | TF-IDF, Random Forest, evaluation metrics |
| joblib | >= 1.3.0 | Model serialization |
| nltk | >= 3.8 | Stopword corpus |
| lime | >= 0.2.0 | LIME explanations |
| streamlit | >= 1.30.0 | Web dashboard |
| numpy | >= 1.24.0 | Numerical operations |
| scipy | >= 1.10.0 | Sparse matrix support |
| matplotlib | >= 3.7.0 | Visualization backend |

### 12. Reproduction

```bash
pip install -r requirements.txt
python download_data.py       # Download WELFake dataset from Kaggle
python train_model.py         # Train and evaluate the model
streamlit run app.py          # Launch the dashboard
```

### 13. Limitations

- **Domain specificity** — The model is trained on English-language political and general news. Performance on other domains (science, sports, satire) is untested.
- **Temporal bias** — The training data reflects news from a specific period. The model may not generalize to emerging topics or evolving writing styles.
- **Feature ceiling** — TF-IDF with 5,000 features captures lexical patterns but cannot model semantic meaning, context, or factual consistency.
- **Overfitting risk** — Training accuracy is 100% while test accuracy is 96%, indicating some memorization of training examples.
- **LIME locality** — LIME explanations are locally faithful but may not reflect the model's global decision boundaries.

### 14. References

- Verma, P.K., Agrawal, P., Amorim, I., Prodan, R. (2021). *WELFake: Word Embedding Over Linguistic Features for Fake News Detection.* IEEE Transactions on Computational Social Systems. DOI: 10.1109/TCSS.2021.3068519
- Ribeiro, M.T., Singh, S., Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD 2016.
