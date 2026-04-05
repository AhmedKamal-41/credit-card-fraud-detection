# Credit Card Fraud Detection

## Project Overview

This project builds a machine learning pipeline to detect fraudulent credit card transactions using the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The dataset contains transactions made by European cardholders in September 2013. It is highly imbalanced: only 0.17% of transactions are fraudulent.

## Goals

- Train an **XGBoost** classifier to distinguish fraudulent from legitimate transactions
- Address severe **class imbalance** using **SMOTE** (Synthetic Minority Oversampling Technique) via `imbalanced-learn`
- Explain model predictions with **SHAP** (SHapley Additive exPlanations)
- Track experiments, metrics, and model artifacts with **MLflow**
- Deploy the trained model as a **FastAPI** REST API backed by a **Streamlit** front-end dashboard

## Project Structure

```
fraud-detection/
├── data/
│   ├── raw/          # Original downloaded dataset (creditcard.csv)
│   └── processed/    # Cleaned, resampled, and split data
├── notebooks/        # Exploratory data analysis and prototyping
├── src/              # Core Python modules (preprocessing, training, evaluation)
├── models/           # Serialized model artifacts
├── api/              # FastAPI app and Streamlit dashboard
└── requirements.txt
```

## Key Design Decisions

| Concern | Approach |
|---|---|
| Class imbalance | SMOTE applied only on the training split to avoid data leakage |
| Model | XGBoost with scale_pos_weight tuned to fraud ratio |
| Evaluation | Precision-Recall AUC, F1, and confusion matrix (accuracy is misleading on imbalanced data) |
| Explainability | SHAP waterfall and summary plots per prediction |
| Experiment tracking | MLflow runs stored locally under `mlruns/`; log params, metrics, and the model artifact |
| Serving | FastAPI `/predict` endpoint; Streamlit app for interactive demo |

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle and place at:
#   data/raw/creditcard.csv

# Run training (from src/)
python train.py

# Start the API
uvicorn api.main:app --reload

# Start the Streamlit dashboard
streamlit run api/dashboard.py
```

## MLflow

Start the MLflow UI to compare runs:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.
