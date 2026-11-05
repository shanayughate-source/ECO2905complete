#!/usr/bin/env python3
"""LEAR-style 24-model LASSO forecast for Ontario DART.

Uses the processed DART feature table, same-hour DART lags from D-2, D-7,
and D-14, and 24 separate hour-specific LASSO regressions. Alpha is selected
with expanding-window time-series cross-validation on training data only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "lear"
RESULT_DIR = ROOT / "results" / MODEL_NAME
COMPARISON_DIR = ROOT / "results" / "model_comparison"

DEFAULT_INPUTS = [
    ROOT / "datasets" / "processed" / "dart_historical_features.csv",
    ROOT / "datasets" / "processed" / "dart_training_features_provincial.csv",
    ROOT / "datasets" / "processed" / "dart_training_features.csv",
]
TARGETS = ["target_dart", "dart", "dart_spread", "target"]
DATES = ["market_date", "delivery_date", "delivery_timestamp", "timestamp", "ts_utc", "datetime", "date"]
HOURS = ["hour_ending", "hourending", "he", "hour", "hour_of_day", "delivery_hour", "hour_beginning"]

# Same-delivery-hour realised prices are not valid D-2 inputs. Target lags are
# constructed explicitly below, using actual DART from D-2/D-7/D-14 only.
LEAKAGE_TOKENS = [
    "rt_lmp", "rt_price", "rt_ozp", "real_time_price", "realtime_price",
    "real_time_lmp", "realtime_lmp", "actual_rt", "realized_rt", "realised_rt",
    "actual_real_time", "realized_real_time", "realised_real_time",
    "da_lmp", "da_price", "da_ozp", "day_ahead_price", "actual_da",
    "settlement_price", "realized_price", "realised_price", "actual_dart",
    "realized_dart", "realised_dart", "dart_spread",
]
LAG_DAYS = [2, 7, 14]
TRAIN_SHARE = 0.70
VALIDATION_SHARE = 0.15
ALPHAS = np.logspace(-3, 2, 25)


def find_col(columns: list[str], aliases: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): str(c) for c in columns}
    return next((lookup[a] for a in aliases if a in lookup), None)


def resolve_input(arg: str | None) -> Path:
    if arg:
        path = Path(arg).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Input CSV not found: {path}")
        return path
    for path in DEFAULT_INPUTS:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No processed DART feature CSV found. Checked:\n"
        + "\n".join(f"  - {p}" for p in DEFAULT_INPUTS)
        + "\nPass the file with --input."
    )


def make_timestamps(raw: pd.DataFrame, date_col: str, hour_col: str | None) -> pd.Series:
    ts = pd.to_datetime(raw[date_col], errors="coerce")
    if hour_col is not None and ts.notna().any() and ts.dt.hour.fillna(0).eq(0).all():
        h = pd.to_numeric(raw[hour_col], errors="coerce")
        if h.between(1, 24).all():
            ts = ts + pd.to_timedelta(h - 1, unit="h")  # IESO HE 1..24 -> hour 0..23
        elif h.between(0, 23).all():
            ts = ts + pd.to_timedelta(h, unit="h")
    return ts


def load_data(path: Path):
    raw = pd.read_csv(path, low_memory=False)
    if raw.empty:
        raise ValueError(f"Input CSV is empty: {path}")

    target_col = find_col(list(raw.columns), TARGETS)
    date_col = find_col(list(raw.columns), DATES)
    hour_col = find_col(list(raw.columns), HOURS)
    if target_col is None:
        raise ValueError(f"DART target not found. Columns: {list(raw.columns)}")
    if date_col is None:
        raise ValueError(f"Delivery date/timestamp not found. Columns: {list(raw.columns)}")

    ts = make_timestamps(raw, date_col, hour_col)
    y = pd.to_numeric(raw[target_col], errors="coerce")
    keep = ts.notna() & y.notna() & np.isfinite(y)
    raw, ts, y = raw.loc[keep].copy(), ts.loc[keep], y.loc[keep]
    order = ts.sort_values(kind="stable").index
    raw = raw.loc[order].reset_index(drop=True)
    ts = ts.loc[order].reset_index(drop=True)
    y = y.loc[order].reset_index(drop=True).astype(float)

    if ts.duplicated().any():
        raise ValueError(
            f"Found {int(ts.duplicated().sum())} duplicate delivery timestamps. "
            "This script expects one Ontario observation per delivery hour; "
            "filter/aggregate multiple zones before training."
        )

    dates = ts.dt.normalize()
    hours = ts.dt.hour.astype(int)
    X = pd.DataFrame(index=raw.index)
    audit: list[dict[str, Any]] = []
    identifiers = {target_col.lower(), date_col.lower()}
    if hour_col:
        identifiers.add(hour_col.lower())

    for col in raw.columns:
        name, low = str(col), str(col).strip().lower()
        if low in identifiers:
            audit.append({"feature": name, "included": False, "reason": "Target/date/hour identifier."})
            continue
        if any(token in low for token in LEAKAGE_TOKENS) or "dart" in low:
            audit.append({"feature": name, "included": False, "reason": "Realised price or DART field; excluded from ex-ante predictors."})
            continue
        numeric = pd.to_numeric(raw[col], errors="coerce")
        observed = int(raw[col].notna().sum())
        if observed == 0 or numeric.notna().sum() / observed < 0.90:
            audit.append({"feature": name, "included": False, "reason": "Not sufficiently numeric or has no observed values."})
            continue
        X[name] = numeric.replace([np.inf, -np.inf], np.nan)
        audit.append({"feature": name, "included": True, "reason": "Numeric predictor retained; verify it was available at D-2."})

    # Date/calendar effects; one model is fitted per delivery hour.
    dow = dates.dt.day_of_week.astype(float)
    month = dates.dt.month.astype(float)
    doy = dates.dt.day_of_year.astype(float)
    calendar = {
        "calendar_dow_sin": np.sin(2 * np.pi * dow / 7),
        "calendar_dow_cos": np.cos(2 * np.pi * dow / 7),
        "calendar_month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "calendar_month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        "calendar_year_sin": np.sin(2 * np.pi * (doy - 1) / 365.25),
        "calendar_year_cos": np.cos(2 * np.pi * (doy - 1) / 365.25),
    }
    for name, values in calendar.items():
        X[name] = np.asarray(values, dtype=float)
        audit.append({"feature": name, "included": True, "reason": "Engineered from delivery calendar."})

    # Exact calendar-date/hour joins prevent a missing hour being mistaken for
    # the previous hour. Lag 2d matches the project's D-2 information cutoff.
    target_index = pd.MultiIndex.from_arrays([dates.to_numpy(), hours.to_numpy()])
    target_lookup = pd.Series(y.to_numpy(), index=target_index)
    for lag in LAG_DAYS:
        lag_dates = dates - pd.to_timedelta(lag, unit="D")
        lag_index = pd.MultiIndex.from_arrays([lag_dates.to_numpy(), hours.to_numpy()])
        name = f"dart_lag_{lag}d_same_hour"
        X[name] = target_lookup.reindex(lag_index).to_numpy(dtype=float)
        audit.append({"feature": name, "included": True, "reason": f"Same-hour observed DART from {lag} days earlier."})

    return X.replace([np.inf, -np.inf], np.nan), y, ts, dates, hours, pd.DataFrame(audit)


def split_dates(dates: pd.Series):
    unique = pd.Index(dates.drop_duplicates().sort_values())
    n = len(unique)
    train_end = int(n * TRAIN_SHARE)
    val_end = int(n * (TRAIN_SHARE + VALIDATION_SHARE))
    if train_end < 100 or val_end <= train_end or n - val_end < 30:
        raise ValueError(f"Insufficient market days for date-based split: {n}")
    return unique[:train_end], unique[train_end:val_end], unique[val_end:]


def pipeline(alpha: float = 1.0) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=alpha, max_iter=50000, tol=1e-4)),
    ])


def fit_hour(Xtr, ytr, Xva, Xtv, ytv):
    # Select features using this hour's training window only.
    missing = Xtr.isna().mean()
    cols = [c for c in Xtr.columns if missing[c] <= 0.40 and Xtr[c].notna().any() and Xtr[c].nunique(dropna=True) > 1]
    if not cols:
        raise ValueError("No usable training predictors for one delivery hour.")

    search = GridSearchCV(
        estimator=pipeline(),
        param_grid={"lasso__alpha": ALPHAS},
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=4, gap=2),
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(Xtr[cols], ytr)
    alpha = float(search.best_params_["lasso__alpha"])
    final = pipeline(alpha)
    final.fit(Xtv[cols], ytv)
    return search.best_estimator_, final, alpha, cols, float(-search.best_score_)


def metrics(y_true, y_pred, threshold: float) -> dict[str, Any]:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    spike = np.abs(actual) >= threshold
    predicted_spike = np.abs(pred) >= threshold
    return {
        "model": MODEL_NAME,
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "r2": float(r2_score(actual, pred)),
        "bias": float(np.mean(pred - actual)),
        "direction_accuracy": float(np.mean(np.sign(actual) == np.sign(pred))),
        "spike_threshold_abs_dart": float(threshold),
        "spike_n": int(spike.sum()),
        "spike_mae": float(np.abs(actual[spike] - pred[spike]).mean()) if spike.any() else np.nan,
        "spike_recall": float((spike & predicted_spike).sum() / spike.sum()) if spike.any() else np.nan,
        "spike_precision": float((spike & predicted_spike).sum() / predicted_spike.sum()) if predicted_spike.any() else np.nan,
    }


def make_predictions(ts, hour, actual, predicted):
    out = pd.DataFrame({
        "delivery_timestamp": pd.to_datetime(ts).dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "delivery_hour": np.asarray(hour, dtype=int),
        "actual_dart": np.asarray(actual, dtype=float),
        "predicted_dart": np.asarray(predicted, dtype=float),
    })
    out["error"] = out["predicted_dart"] - out["actual_dart"]
    out["absolute_error"] = out["error"].abs()
    out["model"] = MODEL_NAME
    return out.sort_values("delivery_timestamp", kind="stable")


def monthly_results(pred: pd.DataFrame) -> pd.DataFrame:
    temp = pred.copy()
    temp["delivery_timestamp"] = pd.to_datetime(temp["delivery_timestamp"])
    temp["month"] = temp["delivery_timestamp"].dt.to_period("M").astype(str)
    rows = []
    for month, group in temp.groupby("month", sort=True):
        rows.append({
            "month": month,
            "n": len(group),
            "mae": mean_absolute_error(group["actual_dart"], group["predicted_dart"]),
            "rmse": np.sqrt(mean_squared_error(group["actual_dart"], group["predicted_dart"])),
            "bias": (group["predicted_dart"] - group["actual_dart"]).mean(),
        })
    return pd.DataFrame(rows)


def update_comparison():
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    names = ["hour_of_week_median", "gaussian_gpd", "lightgbm", "catboost", "xgboost", "lear"]
    rows = []
    for name in names:
        path = ROOT / "results" / name / "test_metrics.csv"
        if not path.exists():
            continue
        try:
            data = pd.read_csv(path)
            if data.empty:
                continue
            row = {str(k).lower(): v for k, v in data.iloc[0].to_dict().items()}
            rows.append({
                "model": name,
                "n": row.get("n", row.get("observations", np.nan)),
                "mae": row.get("mae", np.nan),
                "rmse": row.get("rmse", np.nan),
                "bias": row.get("bias", np.nan),
                "spike_mae": row.get("spike_mae", np.nan),
                "spike_precision": row.get("spike_precision", np.nan),
                "spike_recall": row.get("spike_recall", np.nan),
            })
        except Exception as exc:
            print(f"Warning: could not read {path}: {exc}")
    if rows:
        pd.DataFrame(rows).sort_values("mae", na_position="last").to_csv(
            COMPARISON_DIR / "point_forecast_model_comparison.csv", index=False
        )


def main():
    parser = argparse.ArgumentParser(description="Train LEAR-style LASSO for Ontario DART")
    parser.add_argument("--input", default=None, help="Processed DART feature CSV")
    args = parser.parse_args()

    input_path = resolve_input(args.input)
    X, y, ts, dates, hours, audit = load_data(input_path)
    train_dates, val_dates, test_dates = split_dates(dates)
    train_mask = dates.isin(train_dates)
    val_mask = dates.isin(val_dates)
    test_mask = dates.isin(test_dates)

    lag_cols = [f"dart_lag_{lag}d_same_hour" for lag in LAG_DAYS]
    eligible = X[lag_cols].notna().all(axis=1)
    train_mask &= eligible
    val_mask &= eligible
    test_mask &= eligible
    if train_mask.sum() < 24 * 100 or val_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Too few observations after constructing D-2/D-7/D-14 lags.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(RESULT_DIR / "feature_audit.csv", index=False)

    val_parts, test_parts = [], []
    fitted_models, alpha_rows, coef_rows = {}, [], []
    for hour in sorted(hours.unique()):
        hmask = hours.eq(hour)
        tr = train_mask & hmask
        va = val_mask & hmask
        te = test_mask & hmask
        Xtr, ytr = X.loc[tr].reset_index(drop=True), y.loc[tr].reset_index(drop=True)
        Xva, yva = X.loc[va].reset_index(drop=True), y.loc[va].reset_index(drop=True)
        Xte, yte = X.loc[te].reset_index(drop=True), y.loc[te].reset_index(drop=True)
        if min(len(Xtr), len(Xva), len(Xte)) < 10:
            raise ValueError(f"Insufficient observations for delivery hour {hour}.")
        Xtv = pd.concat([Xtr, Xva], ignore_index=True)
        ytv = pd.concat([ytr, yva], ignore_index=True)
        val_model, final_model, alpha, cols, cv_mae = fit_hour(Xtr, ytr, Xva, Xtv, ytv)

        val_hat = val_model.predict(Xva[cols])
        test_hat = final_model.predict(Xte[cols])
        val_idx = dates.index[va]
        test_idx = dates.index[te]
        val_parts.append(make_predictions(ts.loc[val_idx].reset_index(drop=True), hours.loc[val_idx].reset_index(drop=True), yva, val_hat))
        test_parts.append(make_predictions(ts.loc[test_idx].reset_index(drop=True), hours.loc[test_idx].reset_index(drop=True), yte, test_hat))

        fitted_models[int(hour)] = {"pipeline": final_model, "features": cols, "alpha": alpha}
        alpha_rows.append({"delivery_hour": int(hour), "selected_alpha": alpha, "training_cv_mae": cv_mae,
                           "n_train": len(Xtr), "n_validation": len(Xva), "n_test": len(Xte), "n_features": len(cols)})
        for feature, coef in zip(cols, final_model.named_steps["lasso"].coef_):
            coef_rows.append({"delivery_hour": int(hour), "feature": feature,
                              "coefficient_standardized": float(coef), "absolute_coefficient": float(abs(coef))})
        print(f"Hour {hour:02d}: train={len(Xtr):,}, validation={len(Xva):,}, test={len(Xte):,}, alpha={alpha:.6g}")

    val_pred = pd.concat(val_parts, ignore_index=True).sort_values("delivery_timestamp")
    test_pred = pd.concat(test_parts, ignore_index=True).sort_values("delivery_timestamp")
    threshold = float(np.quantile(np.abs(y.loc[train_mask].to_numpy()), 0.95))
    val_metrics = metrics(val_pred["actual_dart"], val_pred["predicted_dart"], threshold)
    test_metrics = metrics(test_pred["actual_dart"], test_pred["predicted_dart"], threshold)

    val_pred.to_csv(RESULT_DIR / "validation_predictions.csv", index=False)
    test_pred.to_csv(RESULT_DIR / "test_predictions.csv", index=False)
    pd.DataFrame([val_metrics]).to_csv(RESULT_DIR / "validation_metrics.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(RESULT_DIR / "test_metrics.csv", index=False)
    monthly_results(test_pred).to_csv(RESULT_DIR / "monthly_test_performance.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(RESULT_DIR / "selected_alpha_by_hour.csv", index=False)

    coef_df = pd.DataFrame(coef_rows)
    importance = coef_df.groupby("feature", as_index=False).agg(
        mean_absolute_coefficient=("absolute_coefficient", "mean"),
        median_absolute_coefficient=("absolute_coefficient", "median"),
        hours_selected=("absolute_coefficient", lambda values: int((values > 1e-10).sum())),
    ).sort_values("mean_absolute_coefficient", ascending=False)
    importance.to_csv(RESULT_DIR / "feature_importance.csv", index=False)

    joblib.dump({"model": MODEL_NAME, "models_by_hour": fitted_models,
                 "lag_days": LAG_DAYS, "forecast_cutoff": "D-2"}, RESULT_DIR / "model.joblib")

    summary = {
        "model": MODEL_NAME,
        "description": "24 hour-specific LASSO regressions (LEAR-style)",
        "input_file": str(input_path),
        "observations_raw": int(len(y)),
        "market_days_raw": int(dates.nunique()),
        "first_delivery_timestamp": str(ts.min()),
        "last_delivery_timestamp": str(ts.max()),
        "train_start": str(train_dates.min()), "train_end": str(train_dates.max()),
        "validation_start": str(val_dates.min()), "validation_end": str(val_dates.max()),
        "test_start": str(test_dates.min()), "test_end": str(test_dates.max()),
        "n_train_after_lags": int(train_mask.sum()),
        "n_validation_after_lags": int(val_mask.sum()),
        "n_test_after_lags": int(test_mask.sum()),
        "n_hour_models": int(len(fitted_models)),
        "lag_days": LAG_DAYS,
        "cv": "4-fold expanding TimeSeriesSplit, 2-observation gap, MAE alpha selection",
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_audit": str(RESULT_DIR / "feature_audit.csv"),
        "warning": "Manually verify every retained input was available at the D-2 forecast origin.",
    }
    with open(RESULT_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)
    update_comparison()

    print("=" * 72)
    print("ONTARIO DART: LEAR-STYLE HOUR-SPECIFIC LASSO")
    print("=" * 72)
    print(f"Input: {input_path}")
    print(f"Raw observations: {len(y):,} | Market days: {dates.nunique():,}")
    print(f"Hour-specific models: {len(fitted_models)}")
    print(f"Train / validation / test after lags: {int(train_mask.sum()):,} / {int(val_mask.sum()):,} / {int(test_mask.sum()):,}")
    print(f"Test MAE:  {test_metrics['mae']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test R2:   {test_metrics['r2']:.4f}")
    print(f"Spike MAE: {test_metrics['spike_mae']:.4f}")
    print(f"Results: {RESULT_DIR}")
    print("Inspect feature_audit.csv before interpreting results as an ex-ante D-2 forecast.")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
