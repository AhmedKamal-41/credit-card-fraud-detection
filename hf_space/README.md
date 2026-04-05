---
title: Credit Card Fraud Detector
emoji: 🔍
colorFrom: green
colorTo: red
sdk: streamlit
sdk_version: 1.56.0
app_file: app.py
pinned: false
license: mit
short_description: XGBoost fraud classifier with live SHAP explanations
---

# Credit Card Fraud Detector

Interactive dashboard for a credit card fraud detection model trained on the
[Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## What it does

- **Fraud probability score** — colour-coded gauge (green / amber / red)
- **Live SHAP waterfall** — explains every prediction in plain feature contributions
- **Threshold explorer** — shows how precision and recall change as you move the decision threshold
- **Transaction sliders** — adjust Amount, Time, and the 5 most important features in real time

## Model

| Property | Value |
|---|---|
| Algorithm | XGBoost (n_estimators=300, max_depth=6, lr=0.05) |
| Training data | 284,807 transactions; 0.17% fraud rate |
| Resampling | SMOTE applied to training fold only |
| AUC-ROC | 0.9753 |
| Operating threshold | 0.9848 (precision ≥ 90%) |
| Top features | V14, V4, V12, is_round_amount, V1 |

## Dataset

The original dataset contains PCA-transformed features V1–V28 (anonymised for
privacy), plus raw `Time` and `Amount`.  Five engineered features are added:
`hour_of_day`, `is_night`, `log_amount`, `is_round_amount`,
`amount_rolling_mean`, `amount_rolling_std`.

## Files

```
app.py           — Streamlit dashboard (self-contained, no MLflow required)
model.pkl        — Trained sklearn Pipeline (StandardScaler + XGBoostClassifier)
pr_curve.npz     — Pre-computed precision/recall/threshold arrays from the test set
requirements.txt — Pinned Python dependencies
```
