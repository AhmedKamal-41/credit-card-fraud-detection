"""
app.py — Self-contained Hugging Face Spaces version of the fraud-detection
         Streamlit dashboard.

No local MLflow server required.  Loads:
  model.pkl      — sklearn Pipeline (StandardScaler + XGBoostClassifier)
  pr_curve.npz   — pre-computed precision / recall / threshold arrays + SHAP base value

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent

FEATURE_COLS = [
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28",
    "hour_of_day","is_night","log_amount","is_round_amount",
    "amount_rolling_mean","amount_rolling_std",
]

# Top-5 features by mean |SHAP| with slider ranges from training data
TOP5 = {
    "V14":             {"min": -19.0, "max":  7.5, "default":  0.06, "step": 0.05},
    "V4":              {"min":  -5.5, "max": 12.5, "default": -0.01, "step": 0.05},
    "V12":             {"min": -19.0, "max":  4.5, "default":  0.13, "step": 0.05},
    "is_round_amount": {"min":  0,    "max":  1,   "default":  0,    "step": 1   },
    "V1":              {"min": -38.0, "max":  2.5, "default":  0.01, "step": 0.05},
}

DEFAULT_THRESHOLD = 0.9848   # optimal threshold (precision >= 90%) from tune.py


# ── Artifact loading (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    with open(_HERE / "model.pkl", "rb") as f:
        pipeline = pickle.load(f)
    scaler   = pipeline.named_steps["scaler"]
    booster  = pipeline.named_steps["clf"].get_booster()
    explainer = shap.TreeExplainer(booster)
    return pipeline, scaler, explainer


@st.cache_data(show_spinner=False)
def load_pr_data():
    data = np.load(_HERE / "pr_curve.npz")
    return data["precision"], data["recall"], data["thresholds"], float(data["base_val"])


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_row(slider_vals: dict, amount: float, time_sec: float) -> pd.DataFrame:
    hour = int(time_sec // 3600 % 24)
    row  = {col: 0.0 for col in FEATURE_COLS}
    row.update(slider_vals)
    row["log_amount"]          = float(np.log1p(amount))
    row["hour_of_day"]         = hour
    row["is_night"]            = int(0 <= hour <= 5)
    row["amount_rolling_mean"] = amount
    row["amount_rolling_std"]  = 0.0
    return pd.DataFrame([row])[FEATURE_COLS]


def predict_and_explain(pipeline, scaler, explainer, X_df):
    prob  = float(pipeline.predict_proba(X_df)[0, 1])
    X_sc  = scaler.transform(X_df)
    sv    = explainer.shap_values(X_sc)
    # shap_values may return array or list depending on shap version
    sv = sv[0] if isinstance(sv, list) else sv[0]
    return prob, sv


def prob_color(p: float) -> str:
    if p < 0.30: return "#2e7d32"
    if p < 0.70: return "#f57c00"
    return "#c62828"


def prob_label(p: float) -> str:
    if p < 0.30: return "LOW RISK"
    if p < 0.70: return "MODERATE"
    return "HIGH RISK"


# ── SHAP waterfall ────────────────────────────────────────────────────────────
def draw_waterfall(shap_vals, feature_vals, feature_names, base_val) -> plt.Figure:
    n_show  = 10
    abs_ord = np.argsort(np.abs(shap_vals))[::-1][:n_show]
    ordered = abs_ord[::-1]

    sv     = shap_vals[ordered]
    fv     = feature_vals[ordered]
    labels = [f"{feature_names[i]} = {feature_vals[i]:.3g}" for i in ordered]

    total      = base_val + shap_vals.sum()
    cumulative = base_val + np.cumsum(sv)
    starts     = np.concatenate([[base_val], cumulative[:-1]])
    colors     = ["#ef5350" if v > 0 else "#42a5f5" for v in sv]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    bars = ax.barh(range(len(sv)), sv, left=starts, color=colors,
                   height=0.55, edgecolor="#0e1117", linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars, sv)):
        sign  = "+" if val > 0 else ""
        x_pos = starts[i] + val + (0.03 if val > 0 else -0.03)
        ax.text(x_pos, i, f"{sign}{val:.3f}", va="center",
                ha="left" if val > 0 else "right", fontsize=8, color="white")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9, color="white")
    ax.tick_params(axis="x", colors="white", labelsize=8)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.spines["bottom"].set_color("#444")
    ax.axvline(base_val, color="#888", linestyle="--", lw=1,
               label=f"Base {base_val:.3f}")
    ax.axvline(total,    color="white", linestyle=":",  lw=1,
               label=f"Output {total:.3f}")
    ax.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e",
              edgecolor="#444", loc="lower right")
    ax.set_title("SHAP Waterfall — Feature Contributions",
                 color="white", fontsize=11, fontweight="bold", pad=8)
    plt.tight_layout()
    return fig


# ── PR threshold chart ────────────────────────────────────────────────────────
def draw_pr_threshold(prec, rec, thr, threshold) -> plt.Figure:
    idx    = min(np.searchsorted(thr, threshold), len(prec) - 2)
    p_at_t = prec[idx]
    r_at_t = rec[idx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.patch.set_facecolor("#0e1117")

    for ax, y_data, ylabel, color, val_at_t in [
        (axes[0], prec[:-1], "Precision", "#42a5f5", p_at_t),
        (axes[1], rec[:-1],  "Recall",    "#ef5350", r_at_t),
    ]:
        ax.set_facecolor("#0e1117")
        ax.plot(thr, y_data, color=color, lw=2)
        ax.axvline(threshold, color="white", linestyle="--", lw=1.4,
                   label=f"t={threshold:.2f}  →  {val_at_t:.3f}")
        ax.scatter([threshold], [val_at_t], color="white", zorder=5, s=60)
        ax.set_xlabel("Threshold", color="white", fontsize=9)
        ax.set_ylabel(ylabel, color="white", fontsize=9)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        ax.tick_params(colors="white", labelsize=8)
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["bottom","left"]].set_color("#444")
        ax.set_title(f"{ylabel} vs Threshold", color="white",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=9, labelcolor="white", facecolor="#1a1a2e", edgecolor="#444")

    plt.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═════════════════════════════════════════════════════════════════════════════
pipeline, scaler, explainer = load_model()
prec, rec, thr, base_val    = load_pr_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Transaction Input")
    st.caption("Adjust values to explore how each feature affects fraud probability.")
    st.divider()

    amount   = st.slider("Amount ($)", 0.0, 5000.0, 50.0, 1.0)
    time_val = st.slider("Time (seconds since first tx)",
                         0.0, 172800.0, 50000.0, 100.0, format="%.0f")
    hour_display = int(time_val // 3600 % 24)
    st.caption(f"Hour of day: **{hour_display:02d}:00 UTC**"
               + (" 🌙 Night flag active" if hour_display <= 5 else ""))

    st.divider()
    st.markdown("**Top-5 SHAP Features**")
    st.caption("These drive the model's predictions most strongly.")

    slider_vals: dict = {}
    for feat, cfg in TOP5.items():
        if feat == "is_round_amount":
            slider_vals[feat] = float(
                st.select_slider("is_round_amount", options=[0, 1],
                                 value=int(cfg["default"]),
                                 help="1 = whole-dollar amount (e.g. $100.00)")
            )
        else:
            slider_vals[feat] = st.slider(
                feat,
                min_value=float(cfg["min"]),
                max_value=float(cfg["max"]),
                value=float(cfg["default"]),
                step=float(cfg["step"]),
            )

    st.divider()
    threshold = st.slider("Decision Threshold", 0.01, 0.99,
                          DEFAULT_THRESHOLD, 0.01,
                          help="Probability cutoff for the fraud flag")

# ── Inference ─────────────────────────────────────────────────────────────────
X_row        = build_row(slider_vals, amount, time_val)
prob, sv     = predict_and_explain(pipeline, scaler, explainer, X_row)
is_fraud     = prob >= threshold

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("# 🔍 Fraud Detection Dashboard")
st.caption(
    "Built with XGBoost trained on the "
    "[Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) "
    "· 284,807 transactions · 0.17% fraud rate · SMOTE resampling · SHAP explanations"
)
st.divider()

col_score, col_meta = st.columns([1, 2], gap="large")

# ── Score card ────────────────────────────────────────────────────────────────
with col_score:
    color = prob_color(prob)
    label = prob_label(prob)
    st.markdown(
        f"""
        <div style="
            background:{color}22;border:2px solid {color};
            border-radius:16px;padding:28px 20px;text-align:center;
        ">
            <div style="font-size:13px;color:#aaa;letter-spacing:2px;margin-bottom:6px;">
                FRAUD PROBABILITY
            </div>
            <div style="font-size:72px;font-weight:900;color:{color};line-height:1;">
                {prob*100:.1f}%
            </div>
            <div style="font-size:18px;font-weight:700;color:{color};
                        margin-top:10px;letter-spacing:3px;">
                {label}
            </div>
            <div style="margin-top:16px;font-size:13px;font-weight:600;
                        color:{'#ef5350' if is_fraud else '#66bb6a'};">
                {'⚠️ FLAGGED AS FRAUD' if is_fraud else '✓ CLEARED'}
                &nbsp;&nbsp;(threshold&nbsp;=&nbsp;{threshold:.2f})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Meta / feature table ──────────────────────────────────────────────────────
with col_meta:
    st.markdown("#### Transaction Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Amount",      f"${amount:,.2f}")
    m2.metric("Hour (UTC)",  f"{hour_display:02d}:00")
    m3.metric("Night Flag",  "Yes 🌙" if hour_display <= 5 else "No ☀️")
    m4.metric("log(Amount)", f"{np.log1p(amount):.3f}")

    st.markdown("#### Top-5 Feature Values")
    feat_df = pd.DataFrame([
        {
            "Feature":   feat,
            "Value":     f"{slider_vals[feat]:.3g}",
            "SHAP":      round(float(sv[FEATURE_COLS.index(feat)]), 4),
            "Direction": "→ fraud" if sv[FEATURE_COLS.index(feat)] > 0 else "→ legit",
        }
        for feat in TOP5
    ])
    st.dataframe(feat_df, hide_index=True, use_container_width=True)

# ── SHAP waterfall ────────────────────────────────────────────────────────────
st.divider()
st.markdown("### SHAP Explanation — Live Waterfall")
st.caption(
    "**Red** bars push toward fraud · **Blue** bars push away · "
    "Dashed = base rate · Dotted = current prediction"
)
wf_fig = draw_waterfall(sv, X_row.values[0], FEATURE_COLS, base_val)
st.pyplot(wf_fig, use_container_width=True)
plt.close(wf_fig)

# ── Threshold explorer ────────────────────────────────────────────────────────
st.divider()
st.markdown("### Threshold Explorer — Precision & Recall on Test Set")
st.caption(
    "Pre-computed on 56,962 held-out transactions (98 fraud). "
    "Drag the **Decision Threshold** slider in the sidebar."
)
pr_fig = draw_pr_threshold(prec, rec, thr, threshold)
st.pyplot(pr_fig, use_container_width=True)
plt.close(pr_fig)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: XGBoost (n_estimators=300, max_depth=6) · "
    "Trained on SMOTE-resampled data (454,902 rows) · "
    f"AUC-ROC = 0.9753 · Threshold = {threshold:.4f}"
)
