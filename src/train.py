"""
train.py — Train 6 classifiers on SMOTE-resampled data, track every run in
           MLflow, and register the best XGBoost model in the Model Registry.

MLflow experiment : "fraud-detection"
MLflow registry   : "fraud-detector"  (alias: "production")

Per-run logging
---------------
  Tags    : model_name
  Params  : all hyperparameters from get_params()
  Metrics : auc_roc, pr_auc, precision, recall, f1, fpr  (all at threshold=0.3)
  Artifact: trained sklearn Pipeline (scaler + estimator) via mlflow.sklearn

Note on MLflow 3.x
------------------
  Model stages (Staging/Production/Archived) were removed in MLflow 3.0.
  This script uses the replacement API: set_registered_model_alias("production").
  The alias "production" is queryable via models:/fraud-detector@production.

Metrics policy
--------------
  Accuracy is never logged — on a 0.17% fraud base rate it is meaningless.
"""

import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow import MlflowClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(__file__)
DATA_DIR     = os.path.join(_HERE, "..", "data")
PLOTS_DIR    = os.path.join(_HERE, "..", "plots")
MLFLOW_URI = "sqlite:///" + Path(_HERE, "..", "mlflow.db").resolve().as_posix()
os.makedirs(PLOTS_DIR, exist_ok=True)

EXPERIMENT   = "fraud-detection"
REGISTRY_NAME = "fraud-detector"
THRESHOLD    = 0.3


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_resampled.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_resampled.csv")).squeeze()
    test    = pd.read_csv(os.path.join(DATA_DIR, "test_engineered.csv"))
    X_test  = test.drop(columns=["Class", "Time", "Amount"])
    y_test  = test["Class"]
    return X_train, y_train, X_test, y_test


# ── Models — each wrapped with a StandardScaler in a Pipeline ─────────────────
# Wrapping in a Pipeline means the logged artifact is self-contained: no
# separate scaler object needs to be saved or loaded alongside the model.
def build_pipelines():
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
            ("clf", XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, n_jobs=-1,
            )),
        ]),
        "LightGBM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=63,
                random_state=42, n_jobs=-1, verbose=-1,
            )),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=200,
                early_stopping=True, random_state=42,
            )),
        ]),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, proba, threshold):
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
    pr_auc  = auc(r_arr, p_arr)
    roc_auc = roc_auc_score(y_true, proba)

    return {
        "auc_roc":   roc_auc,
        "pr_auc":    pr_auc,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "fpr":       fpr,
    }


# ── PR curve plot ─────────────────────────────────────────────────────────────
def plot_pr_curve(name, y_true, proba, pr_auc):
    p_arr, r_arr, thresholds = precision_recall_curve(y_true, proba)
    idx = min(np.searchsorted(thresholds, THRESHOLD), len(p_arr) - 2)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(r_arr, p_arr, lw=2, color="#1565C0",
            label=f"PR curve (AUC = {pr_auc:.4f})")
    ax.scatter(r_arr[idx], p_arr[idx], zorder=5, s=90, color="#E53935",
               label=f"Threshold={THRESHOLD}  (P={p_arr[idx]:.3f}, R={r_arr[idx]:.3f})")
    ax.axhline(y_true.mean(), color="grey", linestyle="--", lw=1,
               label=f"Baseline (fraud rate={y_true.mean():.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=[0, 1], ylim=[0, 1.05])
    ax.set_title(f"Precision-Recall Curve — {name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()

    safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(PLOTS_DIR, f"pr_curve_{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── MLflow: register best XGBoost run and set production alias ────────────────
def register_best_xgboost(results: list[dict]):
    xgb_runs = [r for r in results if r["model"] == "XGBoost"]
    if not xgb_runs:
        print("No XGBoost runs found — skipping registration.")
        return

    best = max(xgb_runs, key=lambda r: r["metrics"]["auc_roc"])
    run_id   = best["run_id"]
    auc_roc  = best["metrics"]["auc_roc"]
    model_uri = f"runs:/{run_id}/model"

    print(f"\nRegistering best XGBoost run  (run_id={run_id}, AUC-ROC={auc_roc:.4f})")
    mv = mlflow.register_model(model_uri=model_uri, name=REGISTRY_NAME)

    client = MlflowClient()
    client.set_registered_model_alias(
        name=REGISTRY_NAME,
        alias="production",
        version=mv.version,
    )
    # Tag the version so it's discoverable without the alias string
    client.set_model_version_tag(REGISTRY_NAME, mv.version, "stage", "production")

    print(f"Registered as '{REGISTRY_NAME}' version {mv.version}")
    print(f"Alias set     : production")
    print(f"Query via     : models:/{REGISTRY_NAME}@production")


# ── Main ──────────────────────────────────────────────────────────────────────
def train_and_evaluate():
    X_train, y_train, X_test, y_test = load_data()
    pipelines = build_pipelines()

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    results = []

    print(f"\n{'Model':<22}  {'AUC-ROC':>8}  {'PR-AUC':>7}  "
          f"{'Precision':>9}  {'Recall':>7}  {'F1':>7}  {'FPR':>7}")
    print("-" * 78)

    for name, pipeline in pipelines.items():
        with mlflow.start_run(run_name=name) as run:

            # ── Train ──────────────────────────────────────────────────────────
            pipeline.fit(X_train, y_train)
            proba = pipeline.predict_proba(X_test)[:, 1]

            # ── Metrics ────────────────────────────────────────────────────────
            m = compute_metrics(y_test.values, proba, THRESHOLD)

            # ── Log to MLflow ──────────────────────────────────────────────────
            mlflow.set_tag("model_name", name)

            # Log every hyperparameter from the estimator step
            params = pipeline.named_steps["clf"].get_params()
            # Flatten any non-serialisable values to strings
            clean_params = {
                k: (str(v) if not isinstance(v, (int, float, str, bool, type(None)))
                    else v)
                for k, v in params.items()
            }
            mlflow.log_params(clean_params)
            mlflow.log_metrics(m)

            # PR curve as a plot artifact
            plot_path = plot_pr_curve(name, y_test, proba, m["pr_auc"])
            mlflow.log_artifact(plot_path, artifact_path="plots")

            # Full pipeline (scaler + model) as the primary artifact
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            results.append({
                "model":   name,
                "run_id":  run.info.run_id,
                "metrics": m,
            })

        print(f"  {name:<22}  {m['auc_roc']:>8.4f}  {m['pr_auc']:>7.4f}  "
              f"{m['precision']:>9.4f}  {m['recall']:>7.4f}  "
              f"{m['f1']:>7.4f}  {m['fpr']:>7.4f}")

    # ── Comparison table ───────────────────────────────────────────────────────
    df = pd.DataFrame([{
        "Model":     r["model"],
        "AUC-ROC":   r["metrics"]["auc_roc"],
        "PR-AUC":    r["metrics"]["pr_auc"],
        "Precision": r["metrics"]["precision"],
        "Recall":    r["metrics"]["recall"],
        "F1":        r["metrics"]["f1"],
        "FPR":       r["metrics"]["fpr"],
        "Run ID":    r["run_id"][:8] + "...",
    } for r in results]).sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
    df.index += 1

    fmt = {c: "{:.4f}".format for c in ["AUC-ROC", "PR-AUC", "Precision",
                                          "Recall", "F1", "FPR"]}
    print("\n" + "=" * 90)
    print(f"FINAL COMPARISON  (experiment='{EXPERIMENT}', threshold={THRESHOLD})")
    print("=" * 90)
    print(df.to_string(formatters=fmt))

    best_overall = df.iloc[0]
    print(f"\nBest by AUC-ROC : {best_overall['Model']}  ({best_overall['AUC-ROC']:.4f})")
    print(f"Best by F1      : {df.sort_values('F1', ascending=False).iloc[0]['Model']}")

    # ── Register best XGBoost ──────────────────────────────────────────────────
    register_best_xgboost(results)

    print(f"\nMLflow UI: cd fraud-detection && mlflow ui --backend-store-uri mlruns/")


if __name__ == "__main__":
    train_and_evaluate()
