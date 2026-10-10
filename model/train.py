"""
AgriIntel — Improved Model Training
====================================
Improvements over baseline:
1. Log-transform target (income) → biggest MAPE reducer
2. Advanced feature engineering
3. Ensemble: LightGBM + XGBoost + CatBoost stacking
4. Optuna hyperparameter tuning for all models
5. Better outlier handling
6. Label encoding for categoricals
7. Interaction features
8. Repayment probability model (secondary model)

HOW TO USE:
    python model/train.py --data your_data.csv --target income_column
"""

import pandas as pd
import numpy as np
import joblib
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import StackingRegressor

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── CONFIG ─────────────────────────────────────────────────────────
RANDOM_STATE  = 42
N_FOLDS       = 5
N_TRIALS      = 50        # Optuna trials per model
MODEL_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── FEATURE ENGINEERING ────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Numeric safety
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # ── Land productivity proxy
    if "land_size" in df.columns and "annual_rainfall" in df.columns:
        df["land_rain_interaction"] = df["land_size"] * df["annual_rainfall"]

    # ── Irrigation efficiency score
    irr_map = {"drip": 1.0, "sprinkler": 0.85, "canal": 0.7,
               "borewell": 0.75, "rainfed": 0.4, "other": 0.5}
    if "irrigation_type" in df.columns:
        df["irrigation_score"] = df["irrigation_type"].str.lower()\
            .map(irr_map).fillna(0.5)

    # ── Ownership score
    own_map = {"owned": 1.0, "leased": 0.6, "rented": 0.5, "share": 0.4}
    if "ownership_type" in df.columns:
        df["ownership_score"] = df["ownership_type"].str.lower()\
            .map(own_map).fillna(0.5)

    # ── Crop risk category
    low_risk  = ["wheat","rice","paddy","maize"]
    high_risk = ["cotton","sugarcane","tobacco","jute"]
    if "crop_type" in df.columns:
        crop = df["crop_type"].str.lower()
        df["crop_risk"] = np.where(crop.isin(low_risk), 0,
                          np.where(crop.isin(high_risk), 2, 1))

    # ── Season encoding
    season_map = {"kharif": 1, "rabi": 2, "zaid": 3,
                  "summer": 3, "winter": 2, "monsoon": 1}
    if "season" in df.columns:
        df["season_code"] = df["season"].str.lower()\
            .map(season_map).fillna(1)

    # ── Land size bins
    if "land_size" in df.columns:
        df["land_category"] = pd.cut(
            df["land_size"],
            bins=[0, 0.5, 1, 2, 5, 10, 999],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(float)

    # ── NPK composite (if available)
    npk_cols = [c for c in ["nitrogen","phosphorus","potassium"] if c in df.columns]
    if len(npk_cols) == 3:
        df["npk_composite"] = (
            df["nitrogen"] * 0.4 +
            df["phosphorus"] * 0.3 +
            df["potassium"] * 0.3
        )

    # ── Soil quality index (pH optimal range 6-7)
    if "soil_ph" in df.columns:
        df["ph_optimal"] = 1 - abs(df["soil_ph"] - 6.5) / 6.5

    return df


def encode_categoricals(df: pd.DataFrame,
                        encoders: dict = None,
                        fit: bool = True) -> tuple:
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if encoders is None:
        encoders = {}

    for col in cat_cols:
        df[col] = df[col].astype(str).str.lower().str.strip()
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le:
                df[col] = df[col].map(
                    dict(zip(le.classes_, le.transform(le.classes_)))
                ).fillna(-1).astype(int)
            else:
                df[col] = 0

    return df, encoders


# ─── OPTUNA TUNING ──────────────────────────────────────────────────
def tune_lgb(X, y, n_trials=N_TRIALS):
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 300, 1500),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 200),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": RANDOM_STATE, "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
        cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X, y, cv=cv,
                                 scoring="neg_mean_absolute_percentage_error")
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgb(X, y, n_trials=N_TRIALS):
    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 300, 1500),
            "learning_rate":   trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth":       trial.suggest_int("max_depth", 3, 10),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": RANDOM_STATE, "verbosity": 0,
        }
        model = xgb.XGBRegressor(**params)
        cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X, y, cv=cv,
                                 scoring="neg_mean_absolute_percentage_error")
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_catboost(X, y, n_trials=N_TRIALS):
    def objective(trial):
        params = {
            "iterations":   trial.suggest_int("iterations", 300, 1500),
            "learning_rate":trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "depth":        trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg":  trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "random_seed": RANDOM_STATE, "verbose": 0,
        }
        model = CatBoostRegressor(**params)
        cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X, y, cv=cv,
                                 scoring="neg_mean_absolute_percentage_error")
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ─── MAIN TRAINING ──────────────────────────────────────────────────
def train(data_path: str, target_col: str = "annual_income"):
    print("=" * 60)
    print("   AgriIntel — Model Training Pipeline")
    print("=" * 60)

    # ── Load data
    print(f"\n[1/8] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"      Shape: {df.shape}")
    print(f"      Target: {target_col}")

    # ── Remove extreme outliers (income > 99th percentile)
    print("\n[2/8] Cleaning outliers...")
    q99 = df[target_col].quantile(0.99)
    q01 = df[target_col].quantile(0.01)
    df = df[(df[target_col] >= q01) & (df[target_col] <= q99)]
    print(f"      After cleaning: {df.shape}")

    # ── Feature engineering
    print("\n[3/8] Engineering features...")
    df = engineer_features(df)

    # ── Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ── Log transform target (KEY improvement for MAPE)
    print("\n[4/8] Log-transforming target variable...")
    y_log = np.log1p(y)

    # ── Encode categoricals
    print("\n[5/8] Encoding categoricals...")
    X, encoders = encode_categoricals(X, fit=True)
    feature_names = X.columns.tolist()

    # ── Baseline MAPE
    print("\n[6/8] Computing baseline MAPE...")
    ridge = Ridge()
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    baseline = cross_val_score(ridge, X, y_log, cv=cv,
                               scoring="neg_mean_absolute_percentage_error")
    print(f"      Ridge baseline MAPE: {-baseline.mean()*100:.1f}%")

    # ── Tune individual models
    print("\n[7/8] Tuning models with Optuna...")

    print("      Tuning LightGBM...")
    lgb_params = tune_lgb(X, y_log)
    lgb_model = lgb.LGBMRegressor(**lgb_params, random_state=RANDOM_STATE, verbose=-1)

    print("      Tuning XGBoost...")
    xgb_params = tune_xgb(X, y_log)
    xgb_model = xgb.XGBRegressor(**xgb_params, random_state=RANDOM_STATE, verbosity=0)

    print("      Tuning CatBoost...")
    cat_params = tune_catboost(X, y_log)
    cat_model = CatBoostRegressor(**cat_params, random_seed=RANDOM_STATE, verbose=0)

    # ── Stacking ensemble
    print("\n[8/8] Building stacking ensemble...")
    ensemble = StackingRegressor(
        estimators=[
            ("lgb", lgb_model),
            ("xgb", xgb_model),
            ("cat", cat_model),
        ],
        final_estimator=Ridge(),
        cv=3,
        n_jobs=-1
    )

    # ── Cross-validate ensemble
    ensemble_scores = cross_val_score(
        ensemble, X, y_log, cv=cv,
        scoring="neg_mean_absolute_percentage_error"
    )
    print(f"      Ensemble MAPE (log-space): {-ensemble_scores.mean()*100:.1f}%")

    # ── Fit final model on all data
    ensemble.fit(X, y_log)

    # ── Compute real MAPE on held-out fold
    real_mapes = []
    for train_idx, val_idx in cv.split(X):
        ensemble.fit(X.iloc[train_idx], y_log.iloc[train_idx])
        preds_log = ensemble.predict(X.iloc[val_idx])
        preds = np.expm1(preds_log)
        real_mape = mean_absolute_percentage_error(y.iloc[val_idx], preds)
        real_mapes.append(real_mape)

    print(f"\n{'='*60}")
    print(f"  FINAL REAL-SPACE MAPE: {np.mean(real_mapes)*100:.1f}%")
    print(f"{'='*60}")

    # ── Refit on full data
    ensemble.fit(X, y_log)

    # ── Train repayment probability model
    print("\n  Training repayment probability model...")
    # WARNING: PROXY LABEL, not real repayment history. It is derived from income,
    # ownership and irrigation, which are also model inputs, so this model is
    # circular. Replace with real loan outcomes (default / on-time) when available.
    income_median = y.median()
    y_repay = (
        (y > income_median).astype(int) * 0.4 +
        (df.get("ownership_score", pd.Series(0.5, index=df.index)) > 0.7).astype(int) * 0.3 +
        (df.get("irrigation_score", pd.Series(0.5, index=df.index)) > 0.6).astype(int) * 0.3
    )
    y_repay = (y_repay > 0.5).astype(int)

    repay_model = lgb.LGBMClassifier(n_estimators=300, random_state=RANDOM_STATE, verbose=-1)
    repay_model.fit(X, y_repay)

    # ── Save everything
    print("\n  Saving artifacts...")
    artifacts = {
        "model":         ensemble,
        "repay_model":   repay_model,
        "encoders":      encoders,
        "feature_names": feature_names,
        "income_median": float(income_median),
        "income_p25":    float(y.quantile(0.25)),
        "income_p75":    float(y.quantile(0.75)),
        "mape":          float(np.mean(real_mapes)),
        "repay_model_is_proxy": True,
    }
    joblib.dump(artifacts, f"{MODEL_DIR}/agriintel_artifacts.pkl")
    print(f"  Saved to {MODEL_DIR}/agriintel_artifacts.pkl")
    print("\n  Done! Ready to deploy.")
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True, help="Path to CSV dataset")
    parser.add_argument("--target", default="annual_income", help="Target column name")
    args = parser.parse_args()
    train(args.data, args.target)
