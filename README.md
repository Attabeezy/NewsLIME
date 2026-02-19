# NewsLIME: Explainable Fake News Detection System

NewsLIME is a fake news detection system that combines a machine learning classifier with LIME (Local Interpretable Model-agnostic Explanations) to provide transparent, word-level reasoning behind each prediction.

## Features

- **Fake News Detection**: Uses a Random Forest classifier on TF-IDF features.
- **Explainability**:  Integrates LIME to highlight which words influenced the prediction (Real vs. Fake).
- **Interactive Dashboard**:  A Streamlit-based web interface for easy testing.
- **Visual Risk Indicators**: Color-coded risk levels based on prediction probability.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/NewsLIME.git
   cd NewsLIME
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:
   ```bash
   python download_data.py
   ```

4. Train the model (this will generate `model.joblib` and `vectorizer.joblib`):
   ```bash
   python train_model.py
   ```

### Running the App

```bash
streamlit run app.py
```

## detailed Documentation

For a deep dive into the architecture, dataset, and model performance, please see the [Technical Report](REPORT.md).
