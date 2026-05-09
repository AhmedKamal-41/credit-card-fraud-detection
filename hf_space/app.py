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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

TOP5 = {
    "V14":             {"min": -19.0, "max":  7.5, "default":  0.06, "step": 0.05},
    "V4":              {"min":  -5.5, "max": 12.5, "default": -0.01, "step": 0.05},
    "V12":             {"min": -19.0, "max":  4.5, "default":  0.13, "step": 0.05},
    "is_round_amount": {"min":  0,    "max":  1,   "default":  0,    "step": 1   },
    "V1":              {"min": -38.0, "max":  2.5, "default":  0.01, "step": 0.05},
}

DEFAULT_THRESHOLD = 0.9848   # optimal threshold (precision >= 90%) from tune.py

PLOTLY_BG    = "#0e1117"
COLOR_FRAUD  = "#ef5350"
COLOR_LEGIT  = "#42a5f5"


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
    data     = np.load(_HERE / "pr_curve.npz", allow_pickle=True)
    base_raw = data["base_val"]
    base_val = float(base_raw.item()) if np.ndim(base_raw) == 0 else float(np.ravel(base_raw)[0])
    return data["precision"], data["recall"], data["thresholds"], base_val


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


@st.cache_data(show_spinner=False, max_entries=256)
def predict_and_explain(_pipeline, _scaler, _explainer, x_tuple):
    X_df = pd.DataFrame([list(x_tuple)], columns=FEATURE_COLS)
    prob = float(_pipeline.predict_proba(X_df)[0, 1])
    X_sc = _scaler.transform(X_df)
    sv   = _explainer.shap_values(X_sc)
    sv   = sv[0] if isinstance(sv, list) else sv[0]
    return prob, np.asarray(sv)


def prob_color(p: float) -> str:
    if p < 0.30: return "#2e7d32"
    if p < 0.70: return "#f57c00"
    return "#c62828"


def prob_label(p: float) -> str:
    if p < 0.30: return "LOW RISK"
    if p < 0.70: return "MODERATE"
    return "HIGH RISK"


# ── Plotly: fraud-score gauge ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=128)
def score_gauge_fig(prob: float, threshold: float):
    color = prob_color(prob)
    label = prob_label(prob)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number=dict(suffix="%", font=dict(size=44, color=color), valueformat=".1f"),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color="#aaa", size=10),
                      tickcolor="#444"),
            bar=dict(color=color, thickness=0.28),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,  30], color="rgba(46,125,50,0.18)"),
                dict(range=[30, 70], color="rgba(245,124,0,0.18)"),
                dict(range=[70,100], color="rgba(198,40,40,0.18)"),
            ],
            threshold=dict(
                line=dict(color="white", width=3),
                thickness=0.85,
                value=threshold * 100,
            ),
        ),
        title=dict(
            text=f"<span style='font-size:13px;color:#aaa;letter-spacing:2px'>"
                 f"FRAUD PROBABILITY</span><br>"
                 f"<span style='font-size:15px;color:{color};letter-spacing:2px;"
                 f"font-weight:700'>{label}</span>",
            y=0.92,
        ),
        domain=dict(x=[0, 1], y=[0, 0.78]),
    ))

    fig.update_layout(
        paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
        height=260, margin=dict(l=10, r=10, t=70, b=10),
    )
    return fig


# ── Plotly: SHAP waterfall ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=128)
def waterfall_fig(sv_key: tuple, fv_key: tuple, base_val: float):
    sv = np.asarray(sv_key)
    fv = np.asarray(fv_key)

    n_show  = 10
    abs_ord = np.argsort(np.abs(sv))[::-1][:n_show]
    ordered = abs_ord[::-1]

    s_vals = sv[ordered]
    labels = [f"{FEATURE_COLS[i]} = {fv[i]:.3g}" for i in ordered]

    cumulative = base_val + np.cumsum(s_vals)
    starts     = np.concatenate([[base_val], cumulative[:-1]]).tolist()
    total      = float(base_val + sv.sum())
    colors     = [COLOR_FRAUD if v > 0 else COLOR_LEGIT for v in s_vals]
    text       = [f"{'+' if v > 0 else ''}{v:.3f}" for v in s_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=s_vals, base=starts,
        orientation="h",
        marker=dict(color=colors, line=dict(color=PLOTLY_BG, width=0.5)),
        text=text, textposition="outside", textfont=dict(color="white", size=11),
        hovertemplate="%{y}<br>SHAP: %{x:+.3f}<extra></extra>",
        showlegend=False,
    ))

    fig.add_vline(x=base_val, line=dict(color="#888", dash="dash", width=1),
                  annotation_text=f"Base {base_val:.3f}",
                  annotation=dict(font=dict(color="#aaa")))
    fig.add_vline(x=total, line=dict(color="white", dash="dot", width=1),
                  annotation_text=f"Output {total:.3f}",
                  annotation=dict(font=dict(color="white")))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
        height=440, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, automargin=True),
        bargap=0.35,
    )
    return fig


# ── Plotly: PR threshold chart ────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=64)
def pr_threshold_fig(threshold_key: float, _prec, _rec, _thr):
    idx    = min(np.searchsorted(_thr, threshold_key), len(_prec) - 2)
    p_at_t = float(_prec[idx])
    r_at_t = float(_rec[idx])

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Precision vs Threshold", "Recall vs Threshold"),
        horizontal_spacing=0.12,
    )

    for col_i, (y_data, color, val_at_t) in enumerate([
        (_prec[:-1], COLOR_LEGIT, p_at_t),
        (_rec[:-1],  COLOR_FRAUD, r_at_t),
    ], start=1):
        fig.add_trace(go.Scatter(
            x=_thr, y=y_data, mode="lines",
            line=dict(color=color, width=2),
            hovertemplate="t=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
            showlegend=False,
        ), row=1, col=col_i)
        fig.add_vline(x=threshold_key,
                      line=dict(color="white", dash="dash", width=1.4),
                      row=1, col=col_i)
        fig.add_trace(go.Scatter(
            x=[threshold_key], y=[val_at_t], mode="markers",
            marker=dict(color="white", size=10),
            hovertemplate=f"t={threshold_key:.3f}<br>val={val_at_t:.3f}<extra></extra>",
            showlegend=False,
        ), row=1, col=col_i)

    fig.update_xaxes(title_text="Threshold", range=[0, 1], showgrid=False, zeroline=False)
    fig.update_yaxes(range=[0, 1.05], showgrid=False, zeroline=False)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
        height=380, margin=dict(l=20, r=20, t=50, b=10),
        showlegend=False,
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═════════════════════════════════════════════════════════════════════════════
pipeline, scaler, explainer = load_model()
prec, rec, thr, base_val    = load_pr_data()

# ── Sidebar (live sliders) ────────────────────────────────────────────────────
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

# Aliases used through the rest of the page
amount_use    = amount
time_use      = time_val
threshold_use = threshold
slider_use    = slider_vals

# ── Inference ─────────────────────────────────────────────────────────────────
X_row    = build_row(slider_use, amount_use, time_use)
x_tuple  = tuple(np.round(X_row.values[0], 6).tolist())
prob, sv = predict_and_explain(pipeline, scaler, explainer, x_tuple)
is_fraud = prob >= threshold_use

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("# 🔍 Fraud Detection Dashboard")
st.caption(
    "Built with XGBoost trained on the "
    "[Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) "
    "· 284,807 transactions · 0.17% fraud rate · SMOTE resampling · SHAP explanations"
)
st.divider()

col_score, col_meta = st.columns([1, 2], gap="large")

# ── Score card (Plotly gauge) ─────────────────────────────────────────────────
with col_score:
    st.plotly_chart(
        score_gauge_fig(round(float(prob), 4), round(float(threshold_use), 2)),
        use_container_width=True,
        config={"displayModeBar": False, "staticPlot": True},
    )
    if is_fraud:
        st.error(f"⚠️  FLAGGED AS FRAUD  ·  threshold = {threshold_use:.2f}")
    else:
        st.success(f"✓  CLEARED  ·  threshold = {threshold_use:.2f}")

# ── Meta / feature table ──────────────────────────────────────────────────────
with col_meta:
    st.markdown("#### Transaction Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Amount",      f"${amount_use:,.2f}")
    m2.metric("Hour (UTC)",  f"{hour_display:02d}:00")
    m3.metric("Night Flag",  "Yes 🌙" if hour_display <= 5 else "No ☀️")
    m4.metric("log(Amount)", f"{np.log1p(amount_use):.3f}")

    st.markdown("#### Top-5 Feature Values")
    feat_df = pd.DataFrame([
        {
            "Feature":   feat,
            "Value":     f"{slider_use[feat]:.3g}",
            "SHAP":      round(float(sv[FEATURE_COLS.index(feat)]), 4),
            "Direction": "→ fraud" if sv[FEATURE_COLS.index(feat)] > 0 else "→ legit",
        }
        for feat in TOP5
    ])
    st.table(feat_df.set_index("Feature"))

# ── SHAP waterfall ────────────────────────────────────────────────────────────
st.divider()
st.markdown("### SHAP Explanation — Waterfall")
st.caption(
    "**Red** bars push toward fraud · **Blue** bars push away · "
    "Dashed = base rate · Dotted = current prediction"
)
st.plotly_chart(
    waterfall_fig(
        tuple(np.round(sv, 5).tolist()),
        tuple(np.round(X_row.values[0], 5).tolist()),
        float(base_val),
    ),
    use_container_width=True,
    config={"displayModeBar": False},
)

# ── Threshold explorer ────────────────────────────────────────────────────────
st.divider()
st.markdown("### Threshold Explorer — Precision & Recall on Test Set")
st.caption(
    "Pre-computed on 56,962 held-out transactions (98 fraud). "
    "Move the **Decision Threshold** slider in the sidebar and click Predict."
)
st.plotly_chart(
    pr_threshold_fig(round(float(threshold_use), 2), prec, rec, thr),
    use_container_width=True,
    config={"displayModeBar": False},
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: XGBoost (n_estimators=300, max_depth=6) · "
    "Trained on SMOTE-resampled data (454,902 rows) · "
    f"AUC-ROC = 0.9753 · Threshold = {threshold_use:.4f}"
)
