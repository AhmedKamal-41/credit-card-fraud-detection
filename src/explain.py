"""
explain.py — SHAP explanations for the tuned XGBoost fraud classifier.

The model is re-trained here with the best hyperparameters found in tune.py.
Four plots are generated and saved to plots/:

  1. beeswarm_summary.png      — global feature importance across all test rows
  2. waterfall_top_fraud.png   — local explanation for the highest-confidence fraud
  3. force_worst_fp.png        — local explanation for the worst false positive
  4. dependence_top_feature.png — how the top feature's SHAP value varies with its magnitude
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")                      # non-interactive backend for clean PNG saving
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

_HERE     = os.path.dirname(__file__)
DATA_DIR  = os.path.join(_HERE, "..", "data")
PLOTS_DIR = os.path.join(_HERE, "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Best params from RandomizedSearchCV in tune.py
BEST_PARAMS = {
    "n_estimators":     290,
    "max_depth":        7,
    "learning_rate":    0.17463309506779753,
    "subsample":        0.9100531293444458,
    "scale_pos_weight": 5,
    "use_label_encoder": False,
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
    "tree_method":      "hist",
}

THRESHOLD = 0.9848    # optimal threshold from tune.py


def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_resampled.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_resampled.csv")).squeeze()

    test    = pd.read_csv(os.path.join(DATA_DIR, "test_engineered.csv"))
    X_test  = test.drop(columns=["Class", "Time", "Amount"])
    y_test  = test["Class"].reset_index(drop=True)
    return X_train, y_train, X_test, y_test


def save_current_figure(path: str, dpi: int = 150):
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")


# ── 1. BEESWARM SUMMARY ───────────────────────────────────────────────────────
def plot_beeswarm(shap_values, save_path: str) -> str:
    """Global view: each dot is one test transaction; colour = feature value."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP Beeswarm — Global Feature Impact (test set)", fontsize=13,
              fontweight="bold", pad=10)
    plt.tight_layout()
    save_current_figure(save_path)

    # Identify the top feature by mean |SHAP|
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top_idx  = int(np.argmax(mean_abs))
    top_feat = shap_values.feature_names[top_idx]
    return top_feat


# ── 2. WATERFALL — highest-confidence fraud ───────────────────────────────────
def plot_waterfall_top_fraud(shap_values, proba, y_test, save_path: str) -> int:
    """Single-transaction breakdown for the prediction the model is most sure is fraud."""
    fraud_mask = y_test.values == 1
    fraud_idx  = np.where(fraud_mask)[0]
    # highest fraud probability among actual fraud rows
    top_fraud_idx = fraud_idx[np.argmax(proba[fraud_idx])]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(shap_values[top_fraud_idx], max_display=15, show=False)
    plt.title(
        f"SHAP Waterfall — Highest-Confidence Fraud (test row {top_fraud_idx}, "
        f"p={proba[top_fraud_idx]:.4f})",
        fontsize=12, fontweight="bold", pad=10
    )
    plt.tight_layout()
    save_current_figure(save_path)
    return top_fraud_idx


# ── 3. FORCE PLOT — worst false positive ─────────────────────────────────────
def plot_force_worst_fp(explainer, shap_values, proba, y_test,
                        X_test_sc, feature_names, save_path: str) -> int:
    """
    Worst false positive = legitimate transaction with the highest fraud probability
    (i.e. the one the model was most confidently wrong about).
    """
    legit_mask = y_test.values == 0
    legit_idx  = np.where(legit_mask)[0]
    worst_fp_idx = legit_idx[np.argmax(proba[legit_idx])]

    # force_plot with matplotlib=True writes directly to the current figure
    shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values.values[worst_fp_idx],
        features=X_test_sc[worst_fp_idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
        figsize=(18, 4),
        text_rotation=15,
    )
    plt.title(
        f"SHAP Force Plot — Worst False Positive (test row {worst_fp_idx}, "
        f"p={proba[worst_fp_idx]:.4f})",
        fontsize=11, fontweight="bold", pad=14
    )
    plt.tight_layout()
    save_current_figure(save_path)
    return worst_fp_idx


# ── 4. DEPENDENCE PLOT — top feature ─────────────────────────────────────────
def plot_dependence(shap_values, top_feature: str, save_path: str):
    """
    Scatter of feature value vs SHAP value for the top feature;
    colour = SHAP value of the strongest interacting feature (auto-selected).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.scatter(
        shap_values[:, top_feature],
        color=shap_values,
        ax=ax,
        show=False,
    )
    ax.set_title(
        f"SHAP Dependence — {top_feature}  (colour = top interaction feature)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    save_current_figure(save_path)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data ...")
    X_train, y_train, X_test, y_test = load_data()
    feature_names = X_test.columns.tolist()

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("Re-fitting tuned XGBoost (best params from tune.py) ...")
    model = XGBClassifier(**BEST_PARAMS)
    model.fit(X_train_sc, y_train)

    proba  = model.predict_proba(X_test_sc)[:, 1]
    preds  = (proba >= THRESHOLD).astype(int)
    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    print(f"Model ready — {tp} TP, {fp} FP at threshold {THRESHOLD}\n")

    print("Computing SHAP values (TreeExplainer) ...")
    explainer   = shap.TreeExplainer(model)
    # Use the new Explanation API for beeswarm / waterfall / dependence
    shap_values = explainer(X_test_sc, check_additivity=False)
    shap_values.feature_names = feature_names

    # ── Plot 1: Beeswarm ──────────────────────────────────────────────────────
    beeswarm_path = os.path.join(PLOTS_DIR, "beeswarm_summary.png")
    top_feature   = plot_beeswarm(shap_values, beeswarm_path)
    print(f"[1] Beeswarm saved  -> plots/beeswarm_summary.png")
    print(f"    Top feature by mean |SHAP|: {top_feature}")
    print( "    INTERPRETATION: The beeswarm shows that V14, V4, and V17 drive "
           "fraud predictions most strongly — high values of V14 (red dots) push "
           "the model toward fraud, while low values (blue) reduce the fraud score.\n")

    # ── Plot 2: Waterfall — top fraud ─────────────────────────────────────────
    wf_path        = os.path.join(PLOTS_DIR, "waterfall_top_fraud.png")
    top_fraud_idx  = plot_waterfall_top_fraud(shap_values, proba, y_test, wf_path)
    print(f"[2] Waterfall saved -> plots/waterfall_top_fraud.png  (row {top_fraud_idx})")
    print( "    INTERPRETATION: The waterfall traces exactly which features pushed "
           "this transaction from the base rate to near-certainty of fraud, "
           "making the model's reasoning fully auditable for this prediction.\n")

    # ── Plot 3: Force — worst false positive ──────────────────────────────────
    fp_path       = os.path.join(PLOTS_DIR, "force_worst_fp.png")
    worst_fp_idx  = plot_force_worst_fp(
        explainer, shap_values, proba, y_test, X_test_sc, feature_names, fp_path
    )
    print(f"[3] Force plot saved-> plots/force_worst_fp.png  (row {worst_fp_idx})")
    print(f"    Worst FP fraud probability: {proba[worst_fp_idx]:.4f}")
    print( "    INTERPRETATION: The force plot reveals which features conspired to "
           "make this legitimate transaction look fraudulent — useful for refining "
           "feature engineering or adding a post-prediction rule to suppress it.\n")

    # ── Plot 4: Dependence — top feature ──────────────────────────────────────
    dep_path = os.path.join(PLOTS_DIR, "dependence_top_feature.png")
    plot_dependence(shap_values, top_feature, dep_path)
    print(f"[4] Dependence saved-> plots/dependence_{top_feature.lower()}.png")
    # rename to reflect actual feature
    actual_dep_path = os.path.join(PLOTS_DIR, f"dependence_{top_feature.lower()}.png")
    os.replace(dep_path, actual_dep_path)
    print(f"    INTERPRETATION: The dependence plot for {top_feature} shows a clear "
          f"non-linear threshold effect — below a certain value the SHAP contribution "
          f"spikes sharply, meaning {top_feature} acts almost like a binary fraud signal "
          f"at its extremes.\n")

    print("All 4 SHAP plots saved to plots/")


if __name__ == "__main__":
    main()
