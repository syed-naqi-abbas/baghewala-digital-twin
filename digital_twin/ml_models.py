"""
ml_models.py — Trains and serves the three predictive models described in
the solution architecture:
  1. Production forecaster       (GradientBoostingRegressor)
  2. Thermal decay predictor     (Ridge polynomial regression)
  3. Rod-floating / failure risk classifier (RandomForestClassifier)

These are trained on the synthetic dataset from data_gen.py. They act as
ML *surrogates* that should approximate the physics simulator's behaviour
and are cross-checked against it — they are not independent field truth.
See README for the real-data retraining step that remains.
"""

from dataclasses import dataclass
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score

PRODUCTION_FEATURES = ["temp_c", "viscosity_cp", "days_since_steam", "spm", "stroke_in"]
THERMAL_FEATURES = ["days_since_steam", "steam_volume_bbl"]
RISK_FEATURES = ["temp_c", "viscosity_cp", "spm", "stroke_in", "rod_floating_risk_ratio"]


@dataclass
class TrainedModels:
    production_model: object
    thermal_model: object
    risk_model: object
    metrics: dict


def train_all(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> TrainedModels:
    metrics = {}

    # --- 1. Production forecaster ---
    X = df[PRODUCTION_FEATURES]
    y = df["oil_rate_bopd_noisy"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
    prod_model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=random_state)
    prod_model.fit(X_tr, y_tr)
    pred = prod_model.predict(X_te)
    metrics["production_mae"] = float(mean_absolute_error(y_te, pred))
    metrics["production_r2"] = float(r2_score(y_te, pred))

    # --- 2. Thermal decay predictor ---
    X = df[THERMAL_FEATURES]
    y = df["temp_c_noisy"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
    thermal_model = make_pipeline(
        StandardScaler(), PolynomialFeatures(degree=3, include_bias=False), Ridge(alpha=1.0)
    )
    thermal_model.fit(X_tr, y_tr)
    pred = thermal_model.predict(X_te)
    metrics["thermal_mae_c"] = float(mean_absolute_error(y_te, pred))
    metrics["thermal_r2"] = float(r2_score(y_te, pred))

    # --- 3. Rod-floating / failure risk classifier ---
    X = df[RISK_FEATURES]
    y = df["failure_event"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    risk_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=random_state, class_weight="balanced")
    risk_model.fit(X_tr, y_tr)
    pred = risk_model.predict(X_te)
    metrics["risk_accuracy"] = float(accuracy_score(y_te, pred))
    metrics["risk_precision"] = float(precision_score(y_te, pred, zero_division=0))
    metrics["risk_recall"] = float(recall_score(y_te, pred, zero_division=0))

    return TrainedModels(prod_model, thermal_model, risk_model, metrics)


def save_models(models: TrainedModels, out_dir: str = "saved_models") -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(models.production_model, f"{out_dir}/production_model.joblib")
    joblib.dump(models.thermal_model, f"{out_dir}/thermal_model.joblib")
    joblib.dump(models.risk_model, f"{out_dir}/risk_model.joblib")


def load_models(in_dir: str = "saved_models") -> TrainedModels:
    production_model = joblib.load(f"{in_dir}/production_model.joblib")
    thermal_model = joblib.load(f"{in_dir}/thermal_model.joblib")
    risk_model = joblib.load(f"{in_dir}/risk_model.joblib")
    return TrainedModels(production_model, thermal_model, risk_model, metrics={})
