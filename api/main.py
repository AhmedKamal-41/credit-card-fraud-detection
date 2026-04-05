"""
api/main.py — FastAPI fraud-detection service.

Endpoints
---------
GET  /health   → model name, version, alias, feature count
POST /predict  → fraud_probability, is_fraud, top 5 SHAP contributors

The model is loaded once at startup from the MLflow Model Registry under the
alias "production".  SHAP values are computed with TreeExplainer on the raw
XGBoost booster (extracted from the pipeline) so that the scaler's transform
is applied to the input before SHAP sees it, keeping explanations consistent
with training.

Run with:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, Field

# ── MLflow setup ──────────────────────────────────────────────────────────────
_HERE    = Path(__file__).parent
_DB_PATH = (_HERE / ".." / "mlflow.db").resolve().as_posix()
TRACKING_URI  = f"sqlite:///{_DB_PATH}"
REGISTRY_NAME = "fraud-detector"
ALIAS         = "production"
THRESHOLD     = 0.9848   # optimal threshold from tune.py (precision ≥ 90%)

# Feature columns the model was trained on (order matters for SHAP)
FEATURE_COLS = [
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28",
    "hour_of_day","is_night","log_amount","is_round_amount",
    "amount_rolling_mean","amount_rolling_std",
]


# ── App state ─────────────────────────────────────────────────────────────────
class _State:
    pipeline:       Any = None   # full sklearn Pipeline
    scaler:         Any = None   # pipeline["scaler"]
    booster:        Any = None   # raw XGBoost booster for TreeExplainer
    explainer:      Any = None   # shap.TreeExplainer
    model_version:  str = ""
    model_name:     str = REGISTRY_NAME


state = _State()


# ── Lifespan: load model once at startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    mv = client.get_model_version_by_alias(REGISTRY_NAME, ALIAS)
    state.model_version = mv.version

    state.pipeline = mlflow.sklearn.load_model(f"models:/{REGISTRY_NAME}@{ALIAS}")
    state.scaler   = state.pipeline.named_steps["scaler"]

    # Extract the underlying XGBoost Booster for TreeExplainer
    xgb_clf          = state.pipeline.named_steps["clf"]
    state.booster    = xgb_clf.get_booster()
    state.explainer  = shap.TreeExplainer(state.booster)

    print(f"Loaded '{REGISTRY_NAME}' v{state.model_version} (@{ALIAS})")
    yield
    # nothing to teardown


app = FastAPI(
    title="Fraud Detection API",
    description="XGBoost fraud classifier with SHAP explanations",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    """
    Raw transaction features.  Engineered columns (hour_of_day, log_amount,
    etc.) can be provided directly or will be derived from Time and Amount
    if omitted.  Rolling features default to the Amount value itself when
    no prior context is available (single-transaction inference).
    """
    # PCA features
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    # Raw fields used to derive engineered features
    Time: float = Field(..., description="Seconds elapsed since first transaction")
    Amount: float = Field(..., ge=0, description="Transaction amount in USD")
    # Engineered features — auto-derived if not supplied
    hour_of_day:          int   | None = Field(None, ge=0, le=23)
    is_night:             int   | None = Field(None, ge=0, le=1)
    log_amount:           float | None = None
    is_round_amount:      int   | None = Field(None, ge=0, le=1)
    amount_rolling_mean:  float | None = None
    amount_rolling_std:   float | None = Field(None, ge=0)


class ShapContributor(BaseModel):
    feature: str
    value:   float       # raw feature value as seen by the model
    shap:    float       # SHAP contribution (positive = pushes toward fraud)


class PredictResponse(BaseModel):
    fraud_probability: float
    is_fraud:          bool
    threshold_used:    float
    top_shap:          list[ShapContributor]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _derive_features(req: TransactionRequest) -> dict:
    """Fill in any engineered features the caller didn't provide."""
    hour = req.hour_of_day if req.hour_of_day is not None else int(req.Time // 3600 % 24)
    return {
        "hour_of_day":         hour,
        "is_night":            req.is_night if req.is_night is not None
                               else int(0 <= hour <= 5),
        "log_amount":          req.log_amount if req.log_amount is not None
                               else float(np.log1p(req.Amount)),
        "is_round_amount":     req.is_round_amount if req.is_round_amount is not None
                               else int(req.Amount % 1 == 0),
        "amount_rolling_mean": req.amount_rolling_mean if req.amount_rolling_mean is not None
                               else req.Amount,
        "amount_rolling_std":  req.amount_rolling_std if req.amount_rolling_std is not None
                               else 0.0,
    }


def _build_feature_row(req: TransactionRequest) -> pd.DataFrame:
    derived = _derive_features(req)
    raw = req.model_dump(exclude={
        "Time", "Amount",
        "hour_of_day", "is_night", "log_amount",
        "is_round_amount", "amount_rolling_mean", "amount_rolling_std",
    })
    row = {**raw, **derived}
    return pd.DataFrame([row])[FEATURE_COLS]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", summary="Model health check")
def health():
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status":        "ok",
        "model_name":    state.model_name,
        "model_version": state.model_version,
        "alias":         ALIAS,
        "n_features":    len(FEATURE_COLS),
        "threshold":     THRESHOLD,
    }


@app.post("/predict", response_model=PredictResponse, summary="Predict fraud probability")
def predict(transaction: TransactionRequest):
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X_df  = _build_feature_row(transaction)
        X_sc  = state.scaler.transform(X_df)     # scale before SHAP + predict
        prob  = float(state.pipeline.predict_proba(X_df)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Feature error: {exc}") from exc

    # SHAP on the scaled input (consistent with training)
    shap_vals = state.explainer.shap_values(X_sc)[0]   # shape: (n_features,)

    # Top 5 contributors by absolute SHAP value
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
    top_shap = [
        ShapContributor(
            feature=FEATURE_COLS[i],
            value=float(X_df.iloc[0, i]),
            shap=float(shap_vals[i]),
        )
        for i in top_idx
    ]

    return PredictResponse(
        fraud_probability=round(prob, 6),
        is_fraud=prob >= THRESHOLD,
        threshold_used=THRESHOLD,
        top_shap=top_shap,
    )
