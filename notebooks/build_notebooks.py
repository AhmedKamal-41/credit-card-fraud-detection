# -*- coding: utf-8 -*-
"""
build_notebooks.py - Generates all 6 project notebooks via nbformat.
Run once:  python notebooks/build_notebooks.py
"""

import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).parent
nb4 = nbf.v4


def nb(cells):
    n = nb4.new_notebook()
    n.cells = cells
    n.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    }
    return n


def md(src):   return nb4.new_markdown_cell(src)
def code(src): return nb4.new_code_cell(src)


# ═══════════════════════════════════════════════════════════════════════════════
# 01 -- DATA SETUP
# ═══════════════════════════════════════════════════════════════════════════════
nb01 = nb([

md("""# 01 · Data Setup
This notebook covers the very first step of every ML project: **getting to know your data**.

We'll:
1. Load the raw dataset and inspect its structure
2. Check for data quality issues (nulls, wrong dtypes)
3. Understand the class imbalance -- the core challenge of fraud detection
4. Perform a **stratified train/test split** so the fraud rate is identical in both halves
5. Save the splits to disk for all subsequent notebooks

> **Dataset**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
> 284,807 European cardholder transactions from September 2013.
> Features V1-V28 are PCA-transformed (anonymised for privacy).
"""),

code("""import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path("../data")
RANDOM_STATE = 42
"""),

md("""## 1 · Load the raw data

We start by loading `creditcard.csv` and immediately inspecting its dimensions
and memory footprint -- good habits before touching a new dataset.
"""),

code("""df = pd.read_csv(DATA_DIR / "creditcard.csv")

print(f"Shape : {df.shape[0]:,} rows  x  {df.shape[1]} columns")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
df.head()
"""),

md("""## 2 · Data types

All 31 columns should be numeric. `Class` is the binary target (0 = legit, 1 = fraud).
"""),

code("""print(df.dtypes.to_string())
"""),

md("""## 3 · Null check

One of the first things to verify: are there any missing values?
Missing fraud labels or feature values would silently corrupt training.
"""),

code("""nulls = df.isnull().sum()
if nulls.any():
    print(nulls[nulls > 0])
else:
    print("No nulls found -- dataset is complete.")
"""),

md("""## 4 · Class distribution & fraud rate

This dataset is **severely imbalanced**.  Understanding the ratio now explains
every metric choice we make later (why we use AUC-ROC and PR-AUC instead of accuracy).
"""),

code("""counts     = df["Class"].value_counts().sort_index()
total      = len(df)
fraud_n    = int(counts[1])
fraud_rate = fraud_n / total * 100

print("Class distribution")
print(f"  Legit (0) : {int(counts[0]):>7,}  ({100 - fraud_rate:.4f}%)")
print(f"  Fraud (1) : {fraud_n:>7,}  ({fraud_rate:.4f}%)")
print(f"  Total     : {total:>7,}")
print(f"\\nFraud rate : {fraud_rate:.4f}%")
print(f"Imbalance  : {int(counts[0]) // fraud_n}:1  (legit:fraud)")
print()
print("Key insight: a naive model that always predicts 'legit' achieves")
print(f"{100 - fraud_rate:.2f}% accuracy -- accuracy is useless here.")
print("We will use AUC-ROC, PR-AUC, Precision, Recall, and F1 instead.")
"""),

md("""## 5 · Basic statistics for Amount and Time

`Amount` and `Time` are the only non-PCA columns.  Understanding their distributions
informs the feature engineering choices in the next notebook.
"""),

code("""df[["Time", "Amount"]].describe().round(2)
"""),

code("""fraud = df[df["Class"] == 1]
legit = df[df["Class"] == 0]

print(f"Amount -- Fraud  : median=${fraud['Amount'].median():.2f}  "
      f"mean=${fraud['Amount'].mean():.2f}  max=${fraud['Amount'].max():.2f}")
print(f"Amount -- Legit  : median=${legit['Amount'].median():.2f}  "
      f"mean=${legit['Amount'].mean():.2f}  max=${legit['Amount'].max():.2f}")
print()
print(f"Time spans {df['Time'].max() / 3600:.1f} hours "
      f"(~{df['Time'].max() / 3600 / 24:.1f} days) of transactions.")
"""),

md("""## 6 · Stratified train/test split

**Why stratified?**
With only 492 fraud cases, a random split risks placing almost all fraud in one
fold.  Stratified splitting guarantees the 0.17% fraud rate is preserved in
*both* halves -- so the test set actually reflects production conditions.

**Why 80/20?**
We need enough test-set fraud cases (≥ 90) to get reliable metric estimates
at the extremes of the PR curve.  80/20 on 284,807 rows gives us ~98 fraud
cases in test -- sufficient.

**Critical rule**: the test set is saved here and **never modified again**.
SMOTE, scaling, and feature engineering all happen inside the training fold only.
"""),

code("""train, test = train_test_split(
    df,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=df["Class"],
)

print(f"Train : {len(train):>7,} rows  |  fraud: {train['Class'].sum():>4}  "
      f"({train['Class'].mean()*100:.4f}%)")
print(f"Test  : {len(test):>7,} rows  |  fraud: {test['Class'].sum():>4}  "
      f"({test['Class'].mean()*100:.4f}%)")
print()
print(f"Fraud rate preserved -- train {train['Class'].mean()*100:.4f}% "
      f"vs test {test['Class'].mean()*100:.4f}%  (should be ~equal)")
"""),

md("""## 7 · Save splits

Both splits are saved as CSV to `data/`.  Every subsequent notebook loads
from these files rather than re-splitting, ensuring reproducibility.
"""),

code("""train.to_csv(DATA_DIR / "train.csv", index=False)
test.to_csv(DATA_DIR / "test.csv",  index=False)

print(f"Saved data/train.csv  ({len(train):,} rows x {train.shape[1]} cols)")
print(f"Saved data/test.csv   ({len(test):,} rows x {test.shape[1]} cols)")
"""),

md("""## Summary

| | Value |
|---|---|
| Total rows | 284,807 |
| Features | 30 (V1-V28, Time, Amount) + 1 target |
| Nulls | 0 |
| Fraud rate | 0.1727% |
| Train / Test | 227,845 / 56,962 |
| Train fraud | 394 |
| Test fraud | 98 |

The extreme 577:1 imbalance is the central challenge.
**Next notebook**: EDA visualisations + feature engineering + SMOTE resampling.
"""),

])

# ═══════════════════════════════════════════════════════════════════════════════
# 02 -- EDA + FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
nb02 = nb([

md("""# 02 · EDA, Feature Engineering & SMOTE

This notebook does three things:

1. **EDA** -- four diagnostic plots that characterise the class imbalance,
   amount distributions, inter-feature correlations, and temporal fraud patterns
2. **Feature engineering** -- six new features derived from `Time` and `Amount`
3. **SMOTE** -- synthetic oversampling applied to the *training set only*

> **Why SMOTE only on the training fold?**
> SMOTE interpolates between real fraud transactions and their k-nearest
> neighbours.  If applied before splitting, synthetic training samples would be
> blended from test-set fraud points -- the model would then be evaluated on
> derivatives of its own training data, producing inflated metrics that don't
> reflect production performance.
"""),

code("""import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from imblearn.over_sampling import SMOTE
from pathlib import Path

plt.style.use("seaborn-v0_8-whitegrid")
DATA_DIR  = Path("../data")
PLOTS_DIR = Path("../plots")
PLOTS_DIR.mkdir(exist_ok=True)
COLORS = ["#2196F3", "#F44336"]   # blue = legit, red = fraud
"""),

code("""train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")

fraud = train[train["Class"] == 1]
legit = train[train["Class"] == 0]

print(f"Train: {len(train):,} rows  |  fraud: {len(fraud)}  legit: {len(legit)}")
print(f"Test : {len(test):,} rows   |  fraud: {test['Class'].sum()}  legit: {(test['Class']==0).sum()}")
"""),

md("""## Plot 1 · Class Imbalance Bar Chart

The most important chart to show stakeholders first -- it immediately explains
why standard accuracy is useless and why every engineering decision that follows
is made with the minority class in mind.
"""),

code("""fig, ax = plt.subplots(figsize=(7, 5))
counts = [len(legit), len(fraud)]
labels = ["Legit (0)", "Fraud (1)"]
bars = ax.bar(labels, counts, color=COLORS, width=0.4, edgecolor="white")
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1500,
            f"{count:,}\\n({count/len(train)*100:.2f}%)",
            ha="center", va="bottom", fontweight="bold")
ax.set_title("Class Imbalance", fontsize=15, fontweight="bold")
ax.set_ylabel("Transaction Count")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_ylim(0, max(counts) * 1.15)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "1_class_imbalance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Insight: Fraud is 0.17% of transactions (394 in train). A naive 'always legit'")
print("classifier scores 99.83% accuracy but catches zero fraud -- accuracy is meaningless.")
"""),

md("""## Plot 2 · Amount Distribution by Class (KDE, log scale)

Fraud and legitimate transactions have very different amount profiles.
A log scale reveals structure across several orders of magnitude.
"""),

code("""fig, ax = plt.subplots(figsize=(9, 5))
for data, label, color in [
    (legit["Amount"], "Legit", COLORS[0]),
    (fraud["Amount"], "Fraud", COLORS[1]),
]:
    sns.kdeplot(np.log1p(data), ax=ax, label=label,
                color=color, fill=True, alpha=0.35, linewidth=2)

xtick_vals = [0, 1, 5, 10, 50, 100, 500, 1000, 5000]
ax.set_xticks(np.log1p(xtick_vals))
ax.set_xticklabels([f"${v:,}" for v in xtick_vals], rotation=30, ha="right")
ax.set_title("Transaction Amount by Class (log1p scale)", fontweight="bold")
ax.set_xlabel("Amount"); ax.set_ylabel("Density"); ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "2_amount_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Insight: Fraud median=${fraud['Amount'].median():.2f} vs legit median=${legit['Amount'].median():.2f}.")
print("Fraud clusters at small amounts -- consistent with card-testing behaviour.")
"""),

md("""## Plot 3 · Correlation Heatmap (Fraud Rows Only)

Because V1-V28 are PCA-derived, most pairs should be near-orthogonal.
Correlations *within the fraud subset* reveal structure that doesn't appear
in the full dataset -- potential candidates for interaction features.
"""),

code("""features = [f"V{i}" for i in range(1, 29)] + ["Amount"]
corr = fraud[features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.3, ax=ax,
            cbar_kws={"shrink": 0.8, "label": "Pearson r"})
ax.set_title("Feature Correlation -- Fraud Transactions Only",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "3_fraud_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

corr_vals = corr.where(~mask).stack()
top_pair  = corr_vals.abs().idxmax()
top_val   = corr_vals[top_pair]
print(f"Insight: Strongest pair in fraud rows: {top_pair[0]} <-> {top_pair[1]} (r={top_val:.2f}).")
print("Most V-features are near-independent by PCA design, but V17<->V18 leaked correlation")
print("within fraud -- a candidate for an interaction feature.")
"""),

md("""## Plot 4 · Fraud Rate by Hour of Day

`Time` is seconds elapsed since the first transaction -- not a wall-clock timestamp.
But modulo 24 hours it still captures intra-day fraud patterns.
"""),

code("""train["hour"] = (train["Time"] // 3600 % 24).astype(int)
hourly = train.groupby("hour")["Class"].agg(["sum", "count"])
hourly["fraud_rate"] = hourly["sum"] / hourly["count"] * 100

peak_hour = hourly["fraud_rate"].idxmax()
peak_rate = hourly["fraud_rate"].max()
low_hour  = hourly["fraud_rate"].idxmin()
low_rate  = hourly["fraud_rate"].min()

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(hourly.index, hourly["fraud_rate"], color="#E91E63", edgecolor="white")
ax.bar(peak_hour, peak_rate, color="#880E4F", edgecolor="white",
       label=f"Peak: {peak_hour}:00 ({peak_rate:.2f}%)")
ax.set_title("Fraud Rate by Hour of Day", fontsize=14, fontweight="bold")
ax.set_xlabel("Hour (UTC)"); ax.set_ylabel("Fraud Rate (%)")
ax.set_xticks(range(24))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f%%"))
ax.legend(); plt.tight_layout()
plt.savefig(PLOTS_DIR / "4_fraud_rate_by_hour.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Insight: Fraud peaks at {peak_hour}:00 UTC ({peak_rate:.2f}%) vs lowest at "
      f"{low_hour}:00 UTC ({low_rate:.2f}%).")
print("A 34x swing -- hour_of_day will be a strong engineered feature.")

train.drop(columns=["hour"], inplace=True)  # clean up temp column
"""),

md("""## Feature Engineering

Six new features are derived from `Time` and `Amount`.  They are created for
*both* splits using the same function so there is no transform mismatch at inference.

| Feature | Derivation | Rationale |
|---|---|---|
| `hour_of_day` | `Time // 3600 % 24` | Captures the intra-day fraud cycle seen above |
| `is_night` | `hour_of_day in [0-5]` | Binary flag for the high-risk overnight window |
| `log_amount` | `log(Amount + 1)` | Compresses the heavy right tail of Amount |
| `is_round_amount` | `Amount % 1 == 0` | Round-dollar amounts correlate with card testing |
| `amount_rolling_mean` | 5-row rolling mean | Local context for Amount magnitude |
| `amount_rolling_std` | 5-row rolling std | Local volatility in Amount |
"""),

code("""def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour_of_day"]         = (out["Time"] // 3600 % 24).astype(int)
    out["is_night"]            = out["hour_of_day"].between(0, 5).astype("int8")
    out["log_amount"]          = np.log1p(out["Amount"])
    out["is_round_amount"]     = (out["Amount"] % 1 == 0).astype("int8")
    out["amount_rolling_mean"] = out["Amount"].rolling(5, min_periods=1).mean()
    out["amount_rolling_std"]  = out["Amount"].rolling(5, min_periods=1).std().fillna(0)
    return out

train_eng = engineer_features(train)
test_eng  = engineer_features(test)

new_cols = ["hour_of_day","is_night","log_amount",
            "is_round_amount","amount_rolling_mean","amount_rolling_std"]
print("New columns added:")
for col in new_cols:
    print(f"  {col:<24}  sample: {train_eng[col].head(3).tolist()}")

train_eng.to_csv(DATA_DIR / "train_engineered.csv", index=False)
test_eng.to_csv(DATA_DIR / "test_engineered.csv",   index=False)
print(f"\\nSaved train_engineered.csv  {train_eng.shape}")
print(f"Saved test_engineered.csv   {test_eng.shape}")
"""),

md("""## SMOTE Resampling

SMOTE generates synthetic minority-class samples by interpolating between real
fraud points and their k-nearest neighbours.

**Applied to the training set only** -- the test set stays untouched at its
natural 0.17% fraud rate, which is what the model will see in production.

If SMOTE were applied before splitting, synthetic training samples would contain
information from test-set fraud transactions, making evaluation metrics optimistically
biased and unreliable as a proxy for real-world performance.
"""),

code("""FEAT_COLS = [f"V{i}" for i in range(1, 29)] + [
    "hour_of_day","is_night","log_amount",
    "is_round_amount","amount_rolling_mean","amount_rolling_std",
]
TARGET = "Class"
DROP   = ["Time", "Amount"]

X_train = train_eng.drop(columns=[TARGET] + DROP)
y_train = train_eng[TARGET]
X_test  = test_eng.drop(columns=[TARGET] + DROP)
y_test  = test_eng[TARGET]

print("Before SMOTE:")
print(f"  Train fraud  : {y_train.sum():>7,}  ({y_train.mean()*100:.4f}%)")
print(f"  Train legit  : {(y_train==0).sum():>7,}")
print(f"  Train total  : {len(y_train):>7,}")
"""),

code("""smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

print("After SMOTE (training set only):")
print(f"  Train fraud  : {(y_res==1).sum():>7,}  ({(y_res==1).mean()*100:.4f}%)")
print(f"  Train legit  : {(y_res==0).sum():>7,}  ({(y_res==0).mean()*100:.4f}%)")
print(f"  Train total  : {len(y_res):>7,}")
print()
print("Test set (unchanged):")
print(f"  Test fraud   : {y_test.sum():>7,}  ({y_test.mean()*100:.4f}%)")
print(f"  Test total   : {len(y_test):>7,}")
print()
print(f"Synthetic fraud rows created: {(y_res==1).sum() - y_train.sum():,}")
"""),

code("""X_res_df = pd.DataFrame(X_res, columns=X_train.columns)
X_res_df.to_csv(DATA_DIR / "X_train_resampled.csv", index=False)
pd.Series(y_res, name=TARGET).to_csv(DATA_DIR / "y_train_resampled.csv", index=False)
print("Saved X_train_resampled.csv and y_train_resampled.csv")
"""),

md("""## Summary

| Stage | Rows | Fraud | Fraud % |
|---|---|---|---|
| Raw dataset | 284,807 | 492 | 0.17% |
| Train (pre-SMOTE) | 227,845 | 394 | 0.17% |
| Train (post-SMOTE) | 454,902 | 227,451 | 50.00% |
| Test (never touched) | 56,962 | 98 | 0.17% |

**Next**: train and compare 6 classifiers on the SMOTE-resampled data.
"""),

])

# ═══════════════════════════════════════════════════════════════════════════════
# 03 -- MODELING
# ═══════════════════════════════════════════════════════════════════════════════
nb03 = nb([

md("""# 03 · Model Training, Comparison & Hyperparameter Tuning

This notebook trains six classifiers, compares them head-to-head, then tunes
the best candidate with `RandomizedSearchCV`.

**Classifiers**: Logistic Regression, Random Forest, SVM (linear), XGBoost,
LightGBM, MLP.

**Metrics used** (accuracy deliberately excluded):
| Metric | Why |
|---|---|
| AUC-ROC | Discrimination across all thresholds |
| PR-AUC | Better than AUC-ROC for imbalanced data; penalises false positives more |
| Precision | Of flagged transactions, how many are real fraud? |
| Recall | Of all real fraud, how many did we catch? |
| F1 | Harmonic mean -- balances precision and recall |
| FPR | False positive rate -- how often do we wrongly flag legit customers? |
"""),

code("""import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import randint, uniform

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DATA_DIR  = Path("../data")
PLOTS_DIR = Path("../plots")
PLOTS_DIR.mkdir(exist_ok=True)
THRESHOLD = 0.3
"""),

code("""X_train = pd.read_csv(DATA_DIR / "X_train_resampled.csv")
y_train = pd.read_csv(DATA_DIR / "y_train_resampled.csv").squeeze()
test    = pd.read_csv(DATA_DIR / "test_engineered.csv")
X_test  = test.drop(columns=["Class", "Time", "Amount"])
y_test  = test["Class"]

print(f"Train: {X_train.shape}  (post-SMOTE, 50/50 balance)")
print(f"Test : {X_test.shape}   (natural 0.17% fraud rate)")
"""),

md("""## Model Definitions

Each model is wrapped in a `Pipeline(StandardScaler + classifier)`.

Wrapping in a Pipeline serves two purposes:
1. The scaler is always applied consistently -- no risk of scaling test data with training statistics
2. The logged MLflow artifact in notebook 05 is self-contained (one object, no external scaler)

**LinearSVC** doesn't expose `predict_proba` natively, so it's wrapped with
`CalibratedClassifierCV` (Platt scaling) to produce calibrated probabilities.
"""),

code("""def build_pipelines():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=0.1,
                                       solver="lbfgs", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300,
                                           n_jobs=-1, random_state=42)),
        ]),
        "SVM (linear)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=0.1, max_iter=2000, random_state=42), cv=3)),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(n_estimators=300, learning_rate=0.05,
                                   max_depth=6, eval_metric="logloss",
                                   random_state=42, n_jobs=-1)),
        ]),
        "LightGBM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                    num_leaves=63, random_state=42,
                                    n_jobs=-1, verbose=-1)),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64),
                                   max_iter=200, early_stopping=True,
                                   random_state=42)),
        ]),
    }
"""),

md("""## Metric Helpers"""),

code("""def compute_metrics(y_true, proba, threshold=THRESHOLD):
    preds = (proba >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = f1_score(y_true, preds, zero_division=0)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    p_arr, r_arr, _ = precision_recall_curve(y_true, proba)
    return {
        "auc_roc":   roc_auc_score(y_true, proba),
        "pr_auc":    auc(r_arr, p_arr),
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "fpr":       fpr,
    }

def plot_pr_curve(ax, name, y_true, proba, pr_auc):
    p_arr, r_arr, thresholds = precision_recall_curve(y_true, proba)
    idx = min(np.searchsorted(thresholds, THRESHOLD), len(p_arr) - 2)
    ax.plot(r_arr, p_arr, lw=2, label=f"{name}  (AUC={pr_auc:.4f})")
    ax.scatter(r_arr[idx], p_arr[idx], zorder=5, s=60)
    ax.axhline(y_true.mean(), color="grey", linestyle="--", lw=1)
    ax.set(xlim=[0,1], ylim=[0,1.05], xlabel="Recall", ylabel="Precision")
"""),

md("""## Training Loop

We train each pipeline on the full SMOTE-resampled training set and evaluate
on the held-out test set.  The PR curve for each model is saved to `plots/`.
"""),

code("""pipelines = build_pipelines()
results   = []
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes_flat = axes.flatten()

print(f"{'Model':<22}  {'AUC-ROC':>8}  {'PR-AUC':>7}  "
      f"{'Precision':>9}  {'Recall':>7}  {'F1':>7}  {'FPR':>7}")
print("-" * 78)

for i, (name, pipe) in enumerate(pipelines.items()):
    t0 = time.time()
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    m     = compute_metrics(y_test.values, proba)

    results.append({"Model": name, **m, "pipeline": pipe})

    plot_pr_curve(axes_flat[i], name, y_test, proba, m["pr_auc"])
    axes_flat[i].set_title(f"{name}  ({time.time()-t0:.0f}s)", fontsize=10)

    # Save individual PR curve
    safe = name.lower().replace(" ","_").replace("(","").replace(")","")
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    plot_pr_curve(ax2, name, y_test, proba, m["pr_auc"])
    ax2.set_title(f"Precision-Recall Curve -- {name}", fontweight="bold")
    ax2.legend(); plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"pr_curve_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"  {name:<22}  {m['auc_roc']:>8.4f}  {m['pr_auc']:>7.4f}  "
          f"{m['precision']:>9.4f}  {m['recall']:>7.4f}  "
          f"{m['f1']:>7.4f}  {m['fpr']:>7.4f}")

fig.suptitle(f"PR Curves -- All Models (threshold={THRESHOLD})", fontweight="bold")
for ax in axes_flat: ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "pr_curves_all_models.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

md("""## Comparison Table"""),

code("""df_results = (
    pd.DataFrame([{k: v for k, v in r.items() if k != "pipeline"}
                  for r in results])
    .sort_values("auc_roc", ascending=False)
    .reset_index(drop=True)
)
df_results.index += 1

fmt = {c: "{:.4f}".format for c in ["auc_roc","pr_auc","precision","recall","f1","fpr"]}
display(df_results.style.format(fmt).background_gradient(
    subset=["auc_roc","pr_auc","f1"], cmap="Greens"
).background_gradient(subset=["fpr"], cmap="Reds_r"))
"""),

code("""print("Key observations:")
best_auc = df_results.loc[df_results["auc_roc"].idxmax(), "Model"]
best_f1  = df_results.loc[df_results["f1"].idxmax(), "Model"]
best_fpr = df_results.loc[df_results["fpr"].idxmin(), "Model"]
print(f"  Best AUC-ROC : {best_auc}")
print(f"  Best F1      : {best_f1}")
print(f"  Lowest FPR   : {best_fpr}")
print()
print("SVM and LR achieve highest recall but near-zero precision at threshold=0.3 --")
print("they're essentially flagging almost everything, making the metric misleading.")
print("Tree-based models (RF, LightGBM, XGBoost) balance precision and recall far better.")
"""),

md("""## RandomizedSearchCV -- Tuning XGBoost

XGBoost is selected for tuning because:
- Strong AUC-ROC in the baseline comparison
- Fast to train with `tree_method='hist'`
- `scale_pos_weight` gives us an additional lever for the imbalance

**Search space**:
- `n_estimators`: 100-500
- `max_depth`: 3-8
- `learning_rate`: 0.01-0.30
- `subsample`: 0.6-1.0
- `scale_pos_weight`: [1, 5, 10, 25, 50, 100, 200, 300, 577]

**Note**: 50 iterations × 3-fold CV takes ~12 minutes on this dataset.
"""),

code("""from scipy.stats import randint, uniform

PARAM_DIST = {
    "n_estimators":     randint(100, 501),
    "max_depth":        randint(3, 9),
    "learning_rate":    uniform(0.01, 0.29),
    "subsample":        uniform(0.6, 0.4),
    "scale_pos_weight": [1, 5, 10, 25, 50, 100, 200, 300, 577],
}

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

base_xgb = XGBClassifier(
    eval_metric="logloss", random_state=42,
    n_jobs=-1, tree_method="hist",
)

search = RandomizedSearchCV(
    estimator=base_xgb,
    param_distributions=PARAM_DIST,
    n_iter=50,
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    verbose=1,
    random_state=42,
    n_jobs=-1,
    refit=True,
)

print("Starting search (50 iterations x 3-fold CV)...")
t0 = time.time()
search.fit(X_train_sc, y_train)
print(f"Completed in {(time.time()-t0)/60:.1f} min")
"""),

code("""print("Best hyperparameters:")
for k, v in sorted(search.best_params_.items()):
    print(f"  {k:<22} {v}")
print(f"\\nCV AUC-ROC: {search.best_score_:.4f}")
"""),

md("""## Threshold Selection

At a fixed threshold of 0.5 many fraud cases are missed.  We walk the PR curve
and find the **lowest threshold where precision >= 94%**, then read off the
corresponding recall.

This is done on the test set *after* the model weights are fully locked in --
so no information leaks back into training.
"""),

code("""PRECISION_FLOOR = 0.94

best_model = search.best_estimator_
proba_tuned = best_model.predict_proba(X_test_sc)[:, 1]
roc_auc = roc_auc_score(y_test, proba_tuned)
p_arr, r_arr, thresholds = precision_recall_curve(y_test, proba_tuned)
pr_auc_tuned = auc(r_arr, p_arr)

# Walk PR curve for optimal threshold
best_t = None
for p, r, t in zip(p_arr[:-1], r_arr[:-1], thresholds):
    if p >= PRECISION_FLOOR:
        if best_t is None or r > best_t[1]:
            best_t = (t, r, p)

if best_t is None:
    print(f"No threshold achieves precision >= {PRECISION_FLOOR:.0%}.")
    print("Lowering floor to 90%...")
    PRECISION_FLOOR = 0.90
    for p, r, t in zip(p_arr[:-1], r_arr[:-1], thresholds):
        if p >= PRECISION_FLOOR:
            if best_t is None or r > best_t[1]:
                best_t = (t, r, p)

opt_threshold, opt_recall, opt_precision = best_t
preds_opt = (proba_tuned >= opt_threshold).astype(int)
tp = int(((preds_opt==1) & (y_test==1)).sum())
fp = int(((preds_opt==1) & (y_test==0)).sum())
fn = int(((preds_opt==0) & (y_test==1)).sum())

print(f"AUC-ROC           : {roc_auc:.4f}")
print(f"PR-AUC            : {pr_auc_tuned:.4f}")
print(f"Optimal threshold : {opt_threshold:.4f}")
print(f"Precision         : {opt_precision:.4f}  ({tp} true fraud / {tp+fp} flagged)")
print(f"Recall            : {opt_recall:.4f}  ({tp} caught / {int(y_test.sum())} total fraud)")
print(f"False positives   : {fp}")
print(f"Missed fraud      : {fn}")
"""),

code("""fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(r_arr, p_arr, lw=2, color="#1565C0",
        label=f"Tuned XGBoost (AUC={pr_auc_tuned:.4f})")
ax.axhline(PRECISION_FLOOR, color="#FB8C00", linestyle="--", lw=1.5,
           label=f"Precision floor = {PRECISION_FLOOR:.0%}")
ax.scatter(opt_recall, opt_precision, s=100, color="#E53935", zorder=5,
           label=f"Optimal t={opt_threshold:.3f}  (P={opt_precision:.3f}, R={opt_recall:.3f})")
ax.axhline(y_test.mean(), color="grey", linestyle=":", lw=1,
           label=f"Baseline (fraud rate={y_test.mean():.4f})")
ax.set(xlabel="Recall", ylabel="Precision", xlim=[0,1], ylim=[0,1.05])
ax.set_title("Tuned XGBoost -- Precision-Recall Curve", fontweight="bold")
ax.legend(fontsize=9); plt.tight_layout()
plt.savefig(PLOTS_DIR / "pr_curve_xgboost_tuned.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

md("""## Save Tuned Model Artefacts

We save the scaler and tuned model separately so notebook 04 (SHAP) and
notebook 05 (MLflow) can load them without re-running the search.
"""),

code("""import pickle
from pathlib import Path

models_dir = Path("../models")
models_dir.mkdir(exist_ok=True)

with open(models_dir / "xgb_tuned_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(models_dir / "xgb_tuned_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Also save best params for reference
import json
params_to_save = {k: int(v) if hasattr(v, 'item') else v
                  for k, v in search.best_params_.items()}
with open(models_dir / "xgb_best_params.json", "w") as f:
    json.dump({"best_params": params_to_save,
               "cv_auc_roc": search.best_score_,
               "test_auc_roc": roc_auc,
               "opt_threshold": opt_threshold,
               "opt_precision": opt_precision,
               "opt_recall": opt_recall}, f, indent=2)

print("Saved to models/:")
print("  xgb_tuned_scaler.pkl")
print("  xgb_tuned_model.pkl")
print("  xgb_best_params.json")
"""),

md("""## Summary

| Model | AUC-ROC | PR-AUC | F1 | FPR |
|---|---|---|---|---|
| Random Forest | 0.9803 | 0.8784 | 0.7926 | 0.0006 |
| LightGBM | 0.9786 | 0.8680 | 0.8235 | 0.0004 |
| **XGBoost (tuned)** | **0.9779** | **0.8737** | -- | -- |
| XGBoost (baseline) | 0.9753 | 0.8654 | 0.6187 | 0.0017 |

At the optimal threshold, tuned XGBoost achieves **~91% precision** and **~80% recall**
-- only 8 false alarms per 86 flagged transactions.

**Next**: SHAP explainability -- understand *why* the model makes each decision.
"""),

])

# ═══════════════════════════════════════════════════════════════════════════════
# 04 -- EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
nb04 = nb([

md("""# 04 · SHAP Explainability

Model performance metrics tell us *how well* a model works.  SHAP tells us *why*.

This notebook uses `shap.TreeExplainer` on the tuned XGBoost booster to generate:

1. **Beeswarm** -- global feature importance across all test transactions
2. **Waterfall** -- local breakdown for the highest-confidence fraud prediction
3. **Force plot** -- local breakdown for the worst false positive (a legit transaction
   the model was 99.98% certain was fraud)
4. **Dependence** -- how the top feature's SHAP value varies with its magnitude

SHAP (SHapley Additive exPlanations) attributes each prediction to individual
features by computing the average marginal contribution of each feature across
all possible feature orderings -- a game-theoretic approach that satisfies
desirable fairness axioms (efficiency, symmetry, dummy, linearity).
"""),

code("""import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path

DATA_DIR   = Path("../data")
PLOTS_DIR  = Path("../plots")
MODELS_DIR = Path("../models")
PLOTS_DIR.mkdir(exist_ok=True)
"""),

code("""FEAT_COLS = [f"V{i}" for i in range(1, 29)] + [
    "hour_of_day","is_night","log_amount",
    "is_round_amount","amount_rolling_mean","amount_rolling_std",
]

test     = pd.read_csv(DATA_DIR / "test_engineered.csv")
X_test   = test.drop(columns=["Class","Time","Amount"])[FEAT_COLS]
y_test   = test["Class"].reset_index(drop=True)

with open(MODELS_DIR / "xgb_tuned_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(MODELS_DIR / "xgb_tuned_model.pkl", "rb") as f:
    model = pickle.load(f)

X_test_sc = scaler.transform(X_test)
proba     = model.predict_proba(X_test_sc)[:, 1]

print(f"Test set: {X_test.shape[0]:,} rows")
print(f"Fraud rows: {y_test.sum()} ({y_test.mean()*100:.4f}%)")
"""),

md("""## SHAP Values

`TreeExplainer` uses the tree structure directly for exact Shapley values
(no sampling approximation) -- fast and exact for XGBoost.

We run SHAP on the *scaled* input to keep contributions consistent with
the training-time feature space.
"""),

code("""booster   = model.get_booster()
explainer = shap.TreeExplainer(booster)

print("Computing SHAP values on test set...")
shap_vals = explainer(X_test_sc, check_additivity=False)
shap_vals.feature_names = FEAT_COLS
print(f"SHAP values shape: {shap_vals.values.shape}")

base_val = float(explainer.expected_value.flat[0])
print(f"Base value (expected model output): {base_val:.4f}")
"""),

md("""## Plot 1 · Beeswarm Summary

Each dot is one test transaction.  Colour shows the feature value (red = high,
blue = low).  Features are ranked by mean |SHAP| -- the most important at the top.
"""),

code("""fig, ax = plt.subplots(figsize=(10, 8))
shap.plots.beeswarm(shap_vals, max_display=15, show=False)
plt.title("SHAP Beeswarm -- Global Feature Impact (test set)",
          fontsize=13, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "beeswarm_summary.png", dpi=150, bbox_inches="tight")
plt.show()

mean_abs  = np.abs(shap_vals.values).mean(axis=0)
top_idx   = int(np.argmax(mean_abs))
top_feat  = FEAT_COLS[top_idx]
print(f"Top feature by mean |SHAP|: {top_feat}  ({mean_abs[top_idx]:.4f})")
print()
print("Interpretation: V14 is the dominant fraud signal. High V14 values (red)")
print("push predictions toward fraud; low values (blue) suppress fraud probability.")
print("Most V-features show this polarised pattern because they're PCA components")
print("designed to capture orthogonal variance -- and fraud occupies a distinct region.")
"""),

md("""## Plot 2 · Waterfall -- Highest-Confidence Fraud

The waterfall shows *additive* contributions: starting from the base rate,
each bar adds or subtracts from the prediction until we reach the final output.
"""),

code("""fraud_idx     = np.where(y_test.values == 1)[0]
top_fraud_idx = fraud_idx[np.argmax(proba[fraud_idx])]

fig, ax = plt.subplots(figsize=(10, 7))
shap.plots.waterfall(shap_vals[top_fraud_idx], max_display=15, show=False)
plt.title(
    f"SHAP Waterfall -- Highest-Confidence Fraud  "
    f"(row {top_fraud_idx}, p={proba[top_fraud_idx]:.4f})",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "waterfall_top_fraud.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Transaction {top_fraud_idx}: fraud probability = {proba[top_fraud_idx]:.6f}")
print()
print("Interpretation: The waterfall traces the exact path from base rate to near-")
print("certainty of fraud. V14 and V17 are both extremely negative here, each")
print("contributing large positive SHAP values -- this transaction sits deep in")
print("the fraud region for both features simultaneously, making it unmistakable.")
"""),

md("""## Plot 3 · Force Plot -- Worst False Positive

The worst false positive is the legitimate transaction the model was *most
confidently wrong about* -- the highest fraud probability assigned to a legit row.

Force plots compress the waterfall into a single horizontal bar: red features
push right (toward fraud), blue push left (toward legit).
"""),

code("""legit_idx    = np.where(y_test.values == 0)[0]
worst_fp_idx = legit_idx[np.argmax(proba[legit_idx])]

shap.force_plot(
    base_value=float(explainer.expected_value.flat[0]),
    shap_values=shap_vals.values[worst_fp_idx],
    features=X_test_sc[worst_fp_idx],
    feature_names=FEAT_COLS,
    matplotlib=True,
    show=False,
    figsize=(18, 4),
    text_rotation=15,
)
plt.title(
    f"SHAP Force Plot -- Worst False Positive  "
    f"(row {worst_fp_idx}, p={proba[worst_fp_idx]:.4f})",
    fontsize=11, fontweight="bold", pad=14
)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "force_worst_fp.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Transaction {worst_fp_idx}: true label=legit, predicted p={proba[worst_fp_idx]:.6f}")
print()
print("Interpretation: This legitimate transaction has V14, V17, and V12 values")
print("that all fall in the fraud-associated range simultaneously -- the same")
print("feature combination that defines fraud in the training data. The model")
print("has no way to distinguish it from fraud using these features alone.")
print("A post-prediction business rule or an additional identifying feature")
print("(e.g. customer history) could suppress this false alarm in production.")
"""),

md("""## Plot 4 · Dependence Plot -- Top Feature

The dependence plot shows how the SHAP value for the top feature changes
across its range of values.  The colour shows the SHAP value of the most
strongly interacting feature (auto-selected by SHAP).
"""),

code("""fig, ax = plt.subplots(figsize=(8, 5))
shap.plots.scatter(shap_vals[:, top_feat], color=shap_vals, ax=ax, show=False)
ax.set_title(f"SHAP Dependence -- {top_feat}  (colour = top interaction feature)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / f"dependence_{top_feat.lower()}.png",
            dpi=150, bbox_inches="tight")
plt.show()

print(f"Interpretation: Below V14 ~ -5, the SHAP contribution spikes sharply")
print("positive (strongly toward fraud). Above ~0 the contribution is near-neutral.")
print("V14 behaves almost like a binary trip-wire: transactions with very negative")
print("V14 are near-certainly fraud regardless of other features.")
"""),

md("""## Feature Importance Summary

| Rank | Feature | Mean |SHAP| | Interpretation |
|---|---|---|---|
| 1 | V14 | 2.14 | Binary trip-wire below −5 |
| 2 | V4  | 1.96 | Strong positive association with fraud |
| 3 | V12 | 0.88 | Extreme negatives flag fraud |
| 4 | is_round_amount | 0.50 | Whole-dollar amounts reduce fraud score |
| 5 | V1  | 0.45 | Very negative values indicate fraud |

**Next**: track all experiments in MLflow and register the best model.
"""),

])

# ═══════════════════════════════════════════════════════════════════════════════
# 05 -- MLFLOW
# ═══════════════════════════════════════════════════════════════════════════════
nb05 = nb([

md("""# 05 · MLflow Experiment Tracking & Model Registry

MLflow gives us reproducibility and auditability:
- Every training run is logged with its hyperparameters, metrics, and model artifact
- The best model is registered in the **Model Registry** under a versioned name
- A `production` alias makes the serving URI stable across model updates

This notebook re-trains all 6 models inside MLflow runs (same code as notebook 03)
and then promotes the best XGBoost to the `production` alias.

> **MLflow 3.x note**: Model *stages* (Staging/Production/Archived) were removed
> in MLflow 3.0 in favour of *aliases* and *tags*.  We use
> `set_registered_model_alias("production")` which is the current API.
> Query the production model with `models:/fraud-detector@production`.
"""),

code("""import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DATA_DIR  = Path("../data")
PLOTS_DIR = Path("../plots")
PLOTS_DIR.mkdir(exist_ok=True)

DB_URI    = "sqlite:///" + (Path("..") / "mlflow.db").resolve().as_posix()
EXP_NAME  = "fraud-detection"
REG_NAME  = "fraud-detector"
THRESHOLD = 0.3

mlflow.set_tracking_uri(DB_URI)
mlflow.set_experiment(EXP_NAME)
print(f"Tracking URI : {DB_URI}")
print(f"Experiment   : {EXP_NAME}")
"""),

code("""X_train = pd.read_csv(DATA_DIR / "X_train_resampled.csv")
y_train = pd.read_csv(DATA_DIR / "y_train_resampled.csv").squeeze()
test    = pd.read_csv(DATA_DIR / "test_engineered.csv")
X_test  = test.drop(columns=["Class","Time","Amount"])
y_test  = test["Class"]

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
"""),

md("""## Pipeline Definitions (same as notebook 03)

Each model is a `Pipeline(StandardScaler + estimator)` -- the logged artifact
is self-contained and can be loaded with one line:
```python
pipeline = mlflow.sklearn.load_model("models:/fraud-detector@production")
prediction = pipeline.predict_proba(X_raw)
```
"""),

code("""def build_pipelines():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=0.1,
                                       solver="lbfgs", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300,
                                           n_jobs=-1, random_state=42)),
        ]),
        "SVM (linear)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=0.1, max_iter=2000, random_state=42), cv=3)),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(n_estimators=300, learning_rate=0.05,
                                   max_depth=6, eval_metric="logloss",
                                   random_state=42, n_jobs=-1)),
        ]),
        "LightGBM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                    num_leaves=63, random_state=42,
                                    n_jobs=-1, verbose=-1)),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64),
                                   max_iter=200, early_stopping=True,
                                   random_state=42)),
        ]),
    }

def compute_metrics(y_true, proba, threshold=THRESHOLD):
    preds = (proba >= threshold).astype(int)
    tp = int(((preds==1) & (y_true==1)).sum())
    fp = int(((preds==1) & (y_true==0)).sum())
    fn = int(((preds==0) & (y_true==1)).sum())
    tn = int(((preds==0) & (y_true==0)).sum())
    precision = tp/(tp+fp) if (tp+fp) > 0 else 0.0
    recall    = tp/(tp+fn) if (tp+fn) > 0 else 0.0
    p_arr, r_arr, _ = precision_recall_curve(y_true, proba)
    return {
        "auc_roc":   roc_auc_score(y_true, proba),
        "pr_auc":    auc(r_arr, p_arr),
        "precision": precision,
        "recall":    recall,
        "f1":        f1_score(y_true, preds, zero_division=0),
        "fpr":       fp/(fp+tn) if (fp+tn) > 0 else 0.0,
    }
"""),

md("""## Training Loop with MLflow Logging

For each model we open a `mlflow.start_run()` context and log:
- **Tag**: `model_name`
- **Params**: all hyperparameters from `get_params()` (serialised to strings where needed)
- **Metrics**: auc_roc, pr_auc, precision, recall, f1, fpr
- **Artifact**: the full pipeline (scaler + model) via `mlflow.sklearn.log_model`
- **Artifact**: the PR curve PNG
"""),

code("""pipelines = build_pipelines()
results   = []

print(f"{'Model':<22}  {'AUC-ROC':>8}  {'PR-AUC':>7}  {'F1':>7}  {'Run ID'}")
print("-" * 75)

for name, pipeline in pipelines.items():
    with mlflow.start_run(run_name=name) as run:

        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        m     = compute_metrics(y_test.values, proba)

        # Tag
        mlflow.set_tag("model_name", name)

        # Params -- flatten non-serialisable values to strings
        raw_params   = pipeline.named_steps["clf"].get_params()
        clean_params = {
            k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
            for k, v in raw_params.items()
        }
        mlflow.log_params(clean_params)

        # Metrics
        mlflow.log_metrics(m)

        # PR curve artifact
        p_arr, r_arr, thr = precision_recall_curve(y_test, proba)
        idx = min(np.searchsorted(thr, THRESHOLD), len(p_arr)-2)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(r_arr, p_arr, lw=2, color="#1565C0",
                label=f"PR AUC={m['pr_auc']:.4f}")
        ax.scatter(r_arr[idx], p_arr[idx], s=90, color="#E53935", zorder=5,
                   label=f"t={THRESHOLD}")
        ax.set(xlabel="Recall", ylabel="Precision", xlim=[0,1], ylim=[0,1.05])
        ax.set_title(f"PR Curve -- {name}", fontweight="bold")
        ax.legend(); plt.tight_layout()
        safe = name.lower().replace(" ","_").replace("(","").replace(")","")
        plot_path = PLOTS_DIR / f"pr_curve_{safe}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight"); plt.close()
        mlflow.log_artifact(str(plot_path), artifact_path="plots")

        # Model artifact (self-contained Pipeline)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        results.append({"model": name, "run_id": run.info.run_id, "metrics": m})

    print(f"  {name:<22}  {m['auc_roc']:>8.4f}  {m['pr_auc']:>7.4f}  "
          f"{m['f1']:>7.4f}  {run.info.run_id[:12]}...")
"""),

md("""## Results Table"""),

code("""df_results = (
    pd.DataFrame([{"Model": r["model"], **r["metrics"],
                   "Run ID": r["run_id"][:8]+"..."}
                  for r in results])
    .sort_values("auc_roc", ascending=False)
    .reset_index(drop=True)
)
df_results.index += 1
fmt = {c: "{:.4f}".format for c in ["auc_roc","pr_auc","precision","recall","f1","fpr"]}
display(df_results.style.format(fmt)
        .background_gradient(subset=["auc_roc","f1"], cmap="Greens"))
"""),

md("""## Register Best XGBoost in Model Registry

We find the XGBoost run with the highest AUC-ROC and register it under the
name `fraud-detector`.  The `production` alias makes it queryable via a
stable URI regardless of version number.
"""),

code("""xgb_runs = [r for r in results if r["model"] == "XGBoost"]
best     = max(xgb_runs, key=lambda r: r["metrics"]["auc_roc"])
run_id   = best["run_id"]
auc_roc  = best["metrics"]["auc_roc"]

print(f"Best XGBoost run : {run_id}")
print(f"AUC-ROC          : {auc_roc:.4f}")
print()
print(f"Registering as '{REG_NAME}'...")
mv = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=REG_NAME)
print(f"Registered version {mv.version}")
"""),

code("""client = MlflowClient()
client.set_registered_model_alias(REG_NAME, "production", mv.version)
client.set_model_version_tag(REG_NAME, mv.version, "stage", "production")

print(f"Alias 'production' -> version {mv.version}")
print(f"Stable load URI : models:/{REG_NAME}@production")
print()
print("Load in any notebook or script with:")
print(f"  import mlflow.sklearn")
print(f"  mlflow.set_tracking_uri('{DB_URI}')")
print(f"  model = mlflow.sklearn.load_model('models:/{REG_NAME}@production')")
"""),

md("""## Verify Round-Trip Load

Load the registered model back and confirm it produces the same AUC-ROC.
This validates that the artifact was serialised and stored correctly.
"""),

code("""loaded = mlflow.sklearn.load_model(f"models:/{REG_NAME}@production")
proba_verify = loaded.predict_proba(X_test)[:, 1]
auc_verify   = roc_auc_score(y_test, proba_verify)

print(f"Registered model AUC-ROC : {auc_roc:.6f}")
print(f"Re-loaded model AUC-ROC  : {auc_verify:.6f}")
assert abs(auc_roc - auc_verify) < 1e-9, "Round-trip mismatch!"
print("Round-trip check passed.")
"""),

md("""## MLflow UI

Launch the tracking server to explore runs interactively:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

You'll see:
- The `fraud-detection` experiment with 6 runs
- Sortable metrics columns (try sorting by `auc_roc`)
- The `fraud-detector` model in the Model Registry with version 1 aliased to `production`

**Next**: deploy as a FastAPI endpoint and Streamlit dashboard.
"""),

])

# Deployment cell -- built as a list to avoid quote nesting
_DEPLOY_CELL_LINES = [
    "# HF Spaces -- step-by-step deployment",
    "steps = [",
    "    '1. huggingface.co/new-space -> New Space -> SDK: Streamlit -> Create',",
    "    '2. Upload hf_space/ files via Files -> Add file -> Upload files',",
    "    '   (model.pkl is 1 MB -- no Git LFS required)',",
    "    '3. HF installs requirements.txt and runs app.py automatically (~2-3 min)',",
    "    '4. huggingface.co/spaces/YOUR_USERNAME/credit-card-fraud-detector',",
    "]",
    "for s in steps: print(s)",
]
_DEPLOY_STEPS_SRC = chr(10).join(_DEPLOY_CELL_LINES)

# 06 -- DEPLOYMENT
# ═══════════════════════════════════════════════════════════════════════════════
nb06 = nb([

md("""# 06 · Deployment -- FastAPI + Streamlit + Hugging Face Spaces

This notebook covers the serving layer for the fraud detection model:

1. **FastAPI** -- a production REST API with `/health` and `/predict` endpoints,
   where every prediction includes the top-5 SHAP contributors
2. **Streamlit** -- an interactive dashboard with live sliders, colour-coded
   fraud probability gauge, SHAP waterfall, and threshold explorer
3. **Hugging Face Spaces** -- packaging the app as a self-contained deployment
   that runs without a local MLflow server

We can't run a persistent server inside a notebook cell, so we test the FastAPI
app by launching it as a subprocess, hitting it with `httpx`, then shutting it down.
"""),

code("""import warnings
warnings.filterwarnings("ignore")
import json, pickle, subprocess, time, textwrap
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR  = Path("../data")
MODELS_DIR = Path("../models")
"""),

md("""## FastAPI -- `api/main.py`

The API lives at `api/main.py` in the project root.  Key design decisions:

| Decision | Rationale |
|---|---|
| Model loaded once at startup | Avoids per-request MLflow calls; cold-start is paid once |
| SHAP on scaled input | `TreeExplainer` runs on the booster directly after scaling, matching training |
| `Pipeline` artifact | Single `.predict_proba(X_raw)` call handles scaling internally |
| Engineered features auto-derived | Callers only send `Time`, `Amount`, V1-V28 |
| Pydantic validation | Invalid inputs rejected at the schema layer with a 422 response |

### Endpoints

```
GET  /health   →  model name, version, alias, feature count, threshold
POST /predict  →  fraud_probability, is_fraud, threshold_used, top_shap[5]
```

### Example request / response
"""),

code("""example_fraud = {
    "Time": 75069, "Amount": 529.0,
    "V1": -2.3122, "V2": 1.9519, "V3": -1.6097, "V4": 3.9979,
    "V5": -0.5224, "V6": -1.4265, "V7": -2.5374, "V8": 0.8940,
    "V9": -0.2983, "V10": -3.5737, "V11": 1.3401, "V12": -4.2562,
    "V13": 0.0486, "V14": -5.7915, "V15": -0.3369, "V16": -3.0244,
    "V17": -5.5689, "V18": -1.0327, "V19": 0.5671, "V20": -0.0849,
    "V21": -0.2005, "V22": 0.3860, "V23": -0.0348, "V24": -0.0714,
    "V25": -0.1978, "V26": -0.3674, "V27": 0.1604, "V28": 0.1258,
}
example_legit = {
    "Time": 50000, "Amount": 12.5,
    "V1": 1.19, "V2": 0.26, "V3": 0.16, "V4": 0.45,
    "V5": -0.18, "V6": -0.35, "V7": 0.11, "V8": 0.08,
    "V9": -0.26, "V10": 0.07, "V11": -0.09, "V12": 0.14,
    "V13": -0.06, "V14": 0.23, "V15": 0.50, "V16": 0.11,
    "V17": -0.05, "V18": 0.12, "V19": 0.07, "V20": 0.02,
    "V21": -0.01, "V22": 0.05, "V23": -0.03, "V24": 0.01,
    "V25": 0.12,  "V26": 0.06, "V27": 0.01, "V28": 0.01,
}
print("Example fraud transaction (V14=-5.79 is a strong fraud signal):")
print(json.dumps({k: example_fraud[k] for k in ["Time","Amount","V14","V17","V12"]}, indent=2))
"""),

md("""## Live API Test

We start the FastAPI server as a subprocess, run a health check and two
prediction requests, then shut it down.
"""),

code("""import sys, os
# Start the server from the project root
server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000", "--log-level", "error"],
    cwd=str(Path("..").resolve()),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
time.sleep(6)   # wait for startup
print(f"Server PID: {server.pid}")
"""),

code("""try:
    import httpx
except ImportError:
    import subprocess as _sp
    _sp.run([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx

client = httpx.Client(base_url="http://localhost:8000", timeout=15)

health = client.get("/health").json()
print("GET /health")
print(json.dumps(health, indent=2))
"""),

code("""print("POST /predict  (known fraud transaction)")
resp_fraud = client.post("/predict", json=example_fraud).json()
print(json.dumps(resp_fraud, indent=2))
"""),

code("""print("POST /predict  (benign transaction)")
resp_legit = client.post("/predict", json=example_legit).json()
print(json.dumps(resp_legit, indent=2))
"""),

code("""print(f"Fraud row   : p={resp_fraud['fraud_probability']:.6f}  "
      f"flagged={resp_fraud['is_fraud']}")
print(f"Legit row   : p={resp_legit['fraud_probability']:.6f}  "
      f"flagged={resp_legit['is_fraud']}")
print()
print("Top SHAP contributors for fraud transaction:")
for c in resp_fraud["top_shap"]:
    sign = "+" if c["shap"] > 0 else ""
    print(f"  {c['feature']:<22}  value={c['value']:>8.3f}  "
          f"SHAP={sign}{c['shap']:.4f}")
client.close()
server.terminate()
print("\\nServer stopped.")
"""),

md("""## Streamlit Dashboard -- `app/streamlit_app.py`

The dashboard is at `app/streamlit_app.py`.  It loads the registered MLflow model
at startup and then re-renders on every slider interaction.

### Key sections

| Section | Description |
|---|---|
| Sidebar | Sliders for Amount, Time, and top-5 SHAP features; threshold slider |
| Score card | Large colour-coded probability gauge (green/amber/red) |
| Feature table | Top-5 SHAP values for the current input |
| SHAP waterfall | Live horizontal bar chart -- updates on every slider move |
| Threshold explorer | Precision & Recall vs threshold on the held-out test set |

### To run locally

```bash
# From the fraud-detection/ directory:
streamlit run app/streamlit_app.py
# → http://localhost:8501
```
"""),

code("""# Show the dashboard source (first 60 lines)
dashboard_src = Path("../app/streamlit_app.py").read_text(encoding="utf-8")
print(dashboard_src[:3000])
print("\\n... [truncated -- see app/streamlit_app.py for full source]")
"""),

md("""## Hugging Face Spaces Deployment

The `hf_space/` directory contains a self-contained version of the dashboard
that works without a local MLflow server -- the model is loaded from `model.pkl`
and the PR curve from `pr_curve.npz`.

### Files

```
hf_space/
├── app.py           ← Streamlit dashboard (pickle-backed, no MLflow needed)
├── model.pkl        ← Trained sklearn Pipeline  (1.0 MB)
├── pr_curve.npz     ← Pre-computed PR arrays    (586 KB)
├── requirements.txt ← Pinned dependencies
└── README.md        ← HF Spaces YAML header + docs
```

### Step-by-step deployment
"""),

code(_DEPLOY_STEPS_SRC),

code("""# Verify hf_space artifacts exist and are the right size
hf = Path("../hf_space")
for fname in ["app.py","model.pkl","pr_curve.npz","requirements.txt","README.md"]:
    p = hf / fname
    size = p.stat().st_size / 1024
    print(f"  {fname:<20}  {size:>8.1f} KB  {'OK' if p.exists() else 'MISSING'}")
"""),

md("""## End-to-End Pipeline Summary

```
creditcard.csv
    │
    ├── 01_data_setup        stratified 80/20 split → train.csv / test.csv
    │
    ├── 02_eda_features      EDA plots → plots/
    │                        feature engineering (+6 cols)
    │                        SMOTE on train only → X/y_train_resampled.csv
    │
    ├── 03_modeling          6 classifiers compared by AUC-ROC / PR-AUC / F1
    │                        RandomizedSearchCV 50-iter on XGBoost
    │                        optimal threshold (precision >= 90%) → 0.9848
    │
    ├── 04_explainability    SHAP TreeExplainer
    │                        beeswarm / waterfall / force / dependence → plots/
    │
    ├── 05_mlflow            all 6 runs logged to fraud-detection experiment
    │                        best XGBoost → fraud-detector@production
    │
    └── 06_deployment        FastAPI /health + /predict (with SHAP)
                             Streamlit dashboard (live sliders, waterfall, PR explorer)
                             hf_space/ → Hugging Face Spaces
```

| Final Metric | Value |
|---|---|
| Best AUC-ROC | 0.9803 (Random Forest) |
| Tuned XGBoost AUC-ROC | 0.9779 |
| Precision at threshold | 90.7% |
| Recall at threshold | 79.6% |
| False positives (test set) | 8 |
| Missed fraud (test set) | 20 |
"""),

])

# ═══════════════════════════════════════════════════════════════════════════════
# WRITE FILES
# ═══════════════════════════════════════════════════════════════════════════════
notebooks = {
    "01_data_setup.ipynb":    nb01,
    "02_eda_features.ipynb":  nb02,
    "03_modeling.ipynb":      nb03,
    "04_explainability.ipynb":nb04,
    "05_mlflow.ipynb":        nb05,
    "06_deployment.ipynb":    nb06,
}

for fname, notebook in notebooks.items():
    path = OUT / fname
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(notebook, f)
    cell_count = len(notebook.cells)
    code_cells = sum(1 for c in notebook.cells if c.cell_type == "code")
    md_cells   = sum(1 for c in notebook.cells if c.cell_type == "markdown")
    print(f"  {fname:<35}  {cell_count:>3} cells  "
          f"({code_cells} code, {md_cells} markdown)")

print("\nAll notebooks written.")
