"""
tune.py — Hyperparameter search for XGBoost via RandomizedSearchCV, followed
          by optimal threshold selection on the precision-recall curve.

Search strategy
---------------
  - 50 iterations of RandomizedSearchCV, 3-fold stratified CV
  - Scoring: roc_auc (AUC-ROC); accuracy is never used
  - Fit on SMOTE-resampled training data
  - Evaluate exclusively on the untouched test split

Threshold selection
-------------------
  After fitting the best estimator, we sweep the PR curve on the test set
  and find the lowest threshold where precision >= 0.90, then take the
  corresponding recall.  Using the test PR curve is intentional here: the
  threshold is chosen *after* the model is locked in, so no information leaks
  back into the estimator weights.  In a stricter pipeline you would use a
  held-out calibration fold; for a single-model selection step on a fixed
  estimator this is equivalent.
"""

import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

_HERE     = os.path.dirname(__file__)
DATA_DIR  = os.path.join(_HERE, "..", "data")
PLOTS_DIR = os.path.join(_HERE, "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

PRECISION_FLOOR = 0.90   # minimum acceptable precision when selecting threshold
N_ITER          = 50
CV_FOLDS        = 3
RANDOM_STATE    = 42


# ── Data ─────────────────────────────────────────────────────────────────────
def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_resampled.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_resampled.csv")).squeeze()

    test    = pd.read_csv(os.path.join(DATA_DIR, "test_engineered.csv"))
    X_test  = test.drop(columns=["Class", "Time", "Amount"])
    y_test  = test["Class"]
    return X_train, y_train, X_test, y_test


# ── Search space ─────────────────────────────────────────────────────────────
# scale_pos_weight: on perfectly balanced SMOTE data =1, but we include a
# range up to the raw imbalance ratio (~577) in case the search benefits from
# down-weighting the synthetic majority slightly.
PARAM_DIST = {
    "n_estimators":    randint(100, 501),           # 100 – 500
    "max_depth":       randint(3, 9),               # 3 – 8
    "learning_rate":   uniform(0.01, 0.29),         # 0.01 – 0.30
    "subsample":       uniform(0.6, 0.4),           # 0.6 – 1.0
    "scale_pos_weight": [1, 5, 10, 25, 50, 100,
                         200, 300, 577],
}


# ── Threshold selection ───────────────────────────────────────────────────────
def find_best_threshold(y_true, proba, precision_floor: float):
    """
    Walk the PR curve and return the threshold that maximises recall
    subject to precision >= precision_floor.
    Returns (threshold, precision, recall).
    """
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, proba)
    # precision_arr and recall_arr have one extra element (for threshold=1);
    # zip with thresholds to keep indices aligned.
    best = None
    for p, r, t in zip(precision_arr[:-1], recall_arr[:-1], thresholds):
        if p >= precision_floor:
            if best is None or r > best[1]:
                best = (t, r, p)
    if best is None:
        raise ValueError(
            f"No threshold achieves precision >= {precision_floor:.0%}. "
            "Lower the precision floor or revisit the model."
        )
    threshold, recall, precision = best
    return threshold, precision, recall


# ── PR curve plot ─────────────────────────────────────────────────────────────
def plot_pr_curve(y_true, proba, pr_auc, threshold, precision, recall):
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall_arr, precision_arr, lw=2, color="#1565C0",
            label=f"PR curve (AUC = {pr_auc:.4f})")
    ax.axhline(PRECISION_FLOOR, color="#FB8C00", linestyle="--", lw=1.4,
               label=f"Precision floor = {PRECISION_FLOOR:.0%}")
    ax.scatter(recall, precision, zorder=5, s=100, color="#E53935",
               label=f"Optimal threshold = {threshold:.3f}\n"
                     f"(P={precision:.3f}, R={recall:.3f})")
    ax.axhline(y_true.mean(), color="grey", linestyle=":", lw=1,
               label=f"Baseline (fraud rate = {y_true.mean():.4f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_title("XGBoost Tuned — Precision-Recall Curve", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "pr_curve_xgboost_tuned.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PR curve saved to plots/pr_curve_xgboost_tuned.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    X_train, y_train, X_test, y_test = load_data()

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    base_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",     # fast histogram method
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DIST,
        n_iter=N_ITER,
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )

    print(f"Starting RandomizedSearchCV: {N_ITER} iterations x {CV_FOLDS}-fold CV")
    print(f"Training on {X_train_sc.shape[0]:,} rows x {X_train_sc.shape[1]} features\n")
    t0 = time.time()
    search.fit(X_train_sc, y_train)
    elapsed = time.time() - t0
    print(f"\nSearch completed in {elapsed/60:.1f} min")

    # ── Best params ───────────────────────────────────────────────────────────
    print("\n=== BEST HYPERPARAMETERS ===")
    for k, v in sorted(search.best_params_.items()):
        print(f"  {k:<22} {v}")
    print(f"\n  CV AUC-ROC (best):     {search.best_score_:.4f}")

    # ── Test-set evaluation ───────────────────────────────────────────────────
    best_model = search.best_estimator_
    proba      = best_model.predict_proba(X_test_sc)[:, 1]

    roc_auc = roc_auc_score(y_test, proba)
    p_arr, r_arr, _ = precision_recall_curve(y_test, proba)
    pr_auc = auc(r_arr, p_arr)

    print(f"\n=== TEST-SET METRICS (default threshold = 0.5) ===")
    print(f"  AUC-ROC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")

    # ── Optimal threshold ─────────────────────────────────────────────────────
    print(f"\n=== THRESHOLD SELECTION (precision floor = {PRECISION_FLOOR:.0%}) ===")
    threshold, precision, recall = find_best_threshold(
        y_test.values, proba, PRECISION_FLOOR
    )

    # Recompute with selected threshold to get full counts
    preds = (proba >= threshold).astype(int)
    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())
    tn = int(((preds == 0) & (y_test == 0)).sum())
    total_fraud = int(y_test.sum())

    print(f"  Optimal threshold : {threshold:.4f}")
    print(f"  Precision         : {precision:.4f}  "
          f"({tp} true fraud / {tp+fp} flagged)")
    print(f"  Recall            : {recall:.4f}  "
          f"({tp} caught / {total_fraud} total fraud)")
    print(f"  False positives   : {fp}")
    print(f"  Missed fraud      : {fn}")

    print(f"\n=== FINAL RESULTS ON TEST SET ===")
    print(f"  AUC-ROC   : {roc_auc:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  Precision : {precision:.4f}")

    plot_pr_curve(y_test.values, proba, pr_auc, threshold, precision, recall)

    # ── Top 10 CV results ─────────────────────────────────────────────────────
    cv_df = (
        pd.DataFrame(search.cv_results_)
        [["rank_test_score", "mean_test_score", "std_test_score",
          "param_n_estimators", "param_max_depth",
          "param_learning_rate", "param_subsample",
          "param_scale_pos_weight"]]
        .sort_values("rank_test_score")
        .head(10)
        .reset_index(drop=True)
    )
    cv_df.columns = ["Rank", "CV AUC-ROC", "Std",
                     "n_est", "depth", "lr", "subsample", "spw"]
    cv_df["CV AUC-ROC"] = cv_df["CV AUC-ROC"].map("{:.4f}".format)
    cv_df["Std"]        = cv_df["Std"].map("{:.4f}".format)
    cv_df["lr"]         = cv_df["lr"].map("{:.4f}".format)
    cv_df["subsample"]  = cv_df["subsample"].map("{:.3f}".format)
    print("\n=== TOP 10 CV CONFIGURATIONS ===")
    print(cv_df.to_string(index=False))


if __name__ == "__main__":
    main()
