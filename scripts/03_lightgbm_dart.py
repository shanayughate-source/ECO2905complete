#!/usr/bin/env python3
"""Leakage-screened LightGBM model for Ontario DART forecasting."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "lightgbm"
RESULT_DIR = ROOT / "results" / MODEL_NAME
COMPARISON_DIR = ROOT / "results" / "model_comparison"

DEFAULT_INPUTS = [
    ROOT / "datasets" / "processed" / "dart_historical_features.csv",
    ROOT / "datasets" / "processed" / "dart_training_features_provincial.csv",
    ROOT / "datasets" / "processed" / "dart_training_features.csv",
]

TARGET_ALIASES = ["target_dart", "dart", "dart_spread", "target", "DART"]
DATE_ALIASES = [
    "market_date", "delivery_date", "delivery_timestamp",
    "timestamp", "ts_utc", "datetime", "date",
]
HOUR_ALIASES = ["hour_ending", "hourending", "he", "hour", "hour_of_day"]

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
RANDOM_SEED = 42

# Same-delivery-hour realised RT price is not known at a day-ahead forecast origin.
# Historical/lagged RT price predictors are retained if clearly marked as lagged.
RT_PRICE_TOKENS = (
    "rt_lmp", "rt_price", "real_time_lmp", "real_time_price",
    "realtime_lmp", "realtime_price", "rt_ozp", "real_time_ozp",
    "lmp_rt", "price_rt",
)
DART_OUTCOME_NAMES = {
    "actual_dart", "realized_dart", "realised_dart", "dart_actual",
    "dart_realized", "dart_realised", "dart_target", "target_dart",
    "dart_spread",
}
LAG_OR_FORECAST_MARKERS = {
    "lag", "lagged", "previous", "prev", "historical", "yesterday",
    "prior", "shift", "forecast", "predicted", "prediction", "projected",
}


def normalize_name(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return value.strip("_")


def find_column(columns: list[str], aliases: list[str]) -> str | None:
    lookup = {normalize_name(c): str(c) for c in columns}
    for alias in aliases:
        if normalize_name(alias) in lookup:
            return lookup[normalize_name(alias)]
    return None


def resolve_input(input_arg: str | None) -> Path:
    if input_arg:
        path = Path(input_arg).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Input feature CSV does not exist: {path}")
        return path
    for path in DEFAULT_INPUTS:
        if path.exists():
            return path
    checked = "\n".join(f"  - {p}" for p in DEFAULT_INPUTS)
    raise FileNotFoundError(
        f"Could not find a processed DART feature CSV. Checked:\n{checked}\n"
        "Pass the correct file with --input."
    )


def leakage_reason(column: str, target_col: str, date_col: str) -> str | None:
    if column == target_col:
        return "target_column"
    if column == date_col:
        return "date_column"

    name = normalize_name(column)
    if name in DART_OUTCOME_NAMES:
        return "realized_or_target_DART"

    # Preserve explicit lags and forecasts, including common lag1 / lag_1 naming.
    tokens = set(name.split("_"))
    has_lag_or_forecast = bool(tokens & LAG_OR_FORECAST_MARKERS)
    has_lag_or_forecast = has_lag_or_forecast or bool(
        re.search(r"(?:lag|prev|shift)[_]?\d+", name)
    )
    if has_lag_or_forecast:
        return None

    # Exclude current RT price columns, including zone-prefixed/suffixed variants.
    for token in RT_PRICE_TOKENS:
        token = normalize_name(token)
        if (
            name == token
            or name.startswith(token + "_")
            or name.endswith("_" + token)
            or ("_" + token + "_") in ("_" + name + "_")
        ):
            return "contemporaneous_realized_RT_price"
    return None


def load_data(path: Path):
    raw = pd.read_csv(path, low_memory=False)
    if raw.empty:
        raise ValueError(f"Input CSV is empty: {path}")

    target_col = find_column(list(raw.columns), TARGET_ALIASES)
    date_col = find_column(list(raw.columns), DATE_ALIASES)
    hour_col = find_column(list(raw.columns), HOUR_ALIASES)
    if target_col is None:
        raise ValueError(
            f"No DART target found. Expected {TARGET_ALIASES}. "
            f"Available columns: {list(raw.columns)}"
        )
    if date_col is None:
        raise ValueError(
            f"No market/delivery date found. Expected {DATE_ALIASES}. "
            f"Available columns: {list(raw.columns)}"
        )

    timestamps = pd.to_datetime(raw[date_col], errors="coerce")
    # A date at midnight + HE 1..24 becomes hourly timestamps 00:00..23:00.
    if hour_col is not None and timestamps.notna().any():
        hours = pd.to_numeric(raw[hour_col], errors="coerce")
        if timestamps.dt.hour.fillna(0).eq(0).all() and hours.notna().all() and hours.between(1, 24).all():
            timestamps = timestamps + pd.to_timedelta(hours - 1, unit="h")

    target = pd.to_numeric(raw[target_col], errors="coerce")
    keep = timestamps.notna() & target.notna() & np.isfinite(target)
    raw = raw.loc[keep].copy()
    timestamps = timestamps.loc[keep]
    target = target.loc[keep]

    order = timestamps.sort_values(kind="stable").index
    raw = raw.loc[order].reset_index(drop=True)
    timestamps = timestamps.loc[order].reset_index(drop=True)
    y = target.loc[order].reset_index(drop=True).astype(float)

    X = pd.DataFrame(index=raw.index)
    audit_rows: list[dict[str, str]] = []
    for column in raw.columns:
        reason = leakage_reason(str(column), target_col, date_col)
        if reason:
            audit_rows.append({"feature": str(column), "decision": "excluded", "reason": reason})
            continue

        numeric = pd.to_numeric(raw[column], errors="coerce")
        original_nonmissing = int(raw[column].notna().sum())
        if original_nonmissing == 0:
            audit_rows.append({"feature": str(column), "decision": "excluded", "reason": "all_values_missing"})
            continue
        numeric_share = float(numeric.notna().sum() / original_nonmissing)
        if numeric_share < 0.90:
            audit_rows.append({"feature": str(column), "decision": "excluded", "reason": f"not_numeric_enough_{numeric_share:.3f}"})
            continue

        X[str(column)] = numeric
        audit_rows.append({
            "feature": str(column),
            "decision": "retained",
            "reason": "numeric_candidate_check_forecast_availability",
        })

    # pandas 4-compatible date accessors.
    hour = timestamps.dt.hour.astype(float)
    dow = timestamps.dt.day_of_week.astype(float)
    month = timestamps.dt.month.astype(float)
    doy = timestamps.dt.day_of_year.astype(float)
    calendar = {
        "time_hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "time_hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "time_dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "time_dow_cos": np.cos(2 * np.pi * dow / 7.0),
        "time_month_sin": np.sin(2 * np.pi * (month - 1) / 12.0),
        "time_month_cos": np.cos(2 * np.pi * (month - 1) / 12.0),
        "time_year_sin": np.sin(2 * np.pi * (doy - 1) / 365.25),
        "time_year_cos": np.cos(2 * np.pi * (doy - 1) / 365.25),
    }
    for name, values in calendar.items():
        X[name] = values
        audit_rows.append({"feature": name, "decision": "retained", "reason": "calendar_feature"})

    usable = [c for c in X.columns if X[c].notna().any() and X[c].nunique(dropna=True) > 1]
    for column in X.columns:
        if column not in usable:
            audit_rows.append({"feature": column, "decision": "excluded", "reason": "all_missing_or_constant"})
    X = X[usable].copy()
    if X.empty:
        raise ValueError("No usable numeric predictors remain after filtering.")

    audit = pd.DataFrame(audit_rows).drop_duplicates("feature", keep="last")
    return X, y, timestamps, usable, audit


def prepare_splits(X: pd.DataFrame, y: pd.Series, timestamps: pd.Series):
    """Split whole market days in chronological 70% / 15% / 15% blocks."""
    market_day = pd.to_datetime(timestamps).dt.normalize()
    unique_days = pd.Index(market_day.drop_duplicates())
    n_days = len(unique_days)
    train_end = int(n_days * TRAIN_FRACTION)
    validation_end = int(n_days * (TRAIN_FRACTION + VALIDATION_FRACTION))
    if train_end < 10 or validation_end <= train_end or n_days - validation_end < 2:
        raise ValueError(f"Not enough market days for chronological split: {n_days}")

    train_days = unique_days[:train_end]
    val_days = unique_days[train_end:validation_end]
    test_days = unique_days[validation_end:]
    train_mask = market_day.isin(train_days)
    val_mask = market_day.isin(val_days)
    test_mask = market_day.isin(test_days)

    X_train_raw, X_val_raw, X_test_raw = X.loc[train_mask].copy(), X.loc[val_mask].copy(), X.loc[test_mask].copy()
    y_train, y_val, y_test = y.loc[train_mask].copy(), y.loc[val_mask].copy(), y.loc[test_mask].copy()
    t_train, t_val, t_test = timestamps.loc[train_mask].copy(), timestamps.loc[val_mask].copy(), timestamps.loc[test_mask].copy()

    # Learn missing-value filtering and medians from training data only.
    missing_rate = X_train_raw.isna().mean()
    columns = [c for c in X.columns if missing_rate[c] <= 0.40]
    X_train_raw, X_val_raw, X_test_raw = X_train_raw[columns], X_val_raw[columns], X_test_raw[columns]
    medians = X_train_raw.median(numeric_only=True)
    columns = [c for c in columns if c in medians.index and pd.notna(medians[c])]
    if not columns:
        raise ValueError("No predictors remain after train-only missing-value filtering.")

    X_train = X_train_raw[columns].fillna(medians[columns])
    X_val = X_val_raw[columns].fillna(medians[columns])
    X_test = X_test_raw[columns].fillna(medians[columns])
    varying = [c for c in columns if X_train[c].nunique(dropna=True) > 1]
    X_train, X_val, X_test = X_train[varying], X_val[varying], X_test[varying]
    if X_train.empty:
        raise ValueError("All predictors are constant in the training period.")

    split_info = {
        "n_market_days": int(n_days),
        "train_market_days": int(len(train_days)),
        "validation_market_days": int(len(val_days)),
        "test_market_days": int(len(test_days)),
        "train_start_day": str(train_days[0].date()),
        "train_end_day": str(train_days[-1].date()),
        "validation_start_day": str(val_days[0].date()),
        "validation_end_day": str(val_days[-1].date()),
        "test_start_day": str(test_days[0].date()),
        "test_end_day": str(test_days[-1].date()),
    }
    return X_train, X_val, X_test, y_train, y_val, y_test, t_train, t_val, t_test, varying, split_info


def calculate_metrics(y_true, y_pred, spike_threshold: float) -> dict[str, Any]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    errors = np.abs(actual - predicted)
    actual_spike = np.abs(actual) >= spike_threshold
    predicted_spike = np.abs(predicted) >= spike_threshold
    return {
        "model": MODEL_NAME,
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(predicted - actual)),
        "direction_accuracy": float(np.mean(np.sign(actual) == np.sign(predicted))),
        "spike_threshold_abs_dart": float(spike_threshold),
        "spike_n": int(actual_spike.sum()),
        "spike_mae": float(errors[actual_spike].mean()) if actual_spike.any() else np.nan,
        "spike_recall": float((actual_spike & predicted_spike).sum() / actual_spike.sum()) if actual_spike.any() else np.nan,
        "spike_precision": float((actual_spike & predicted_spike).sum() / predicted_spike.sum()) if predicted_spike.any() else np.nan,
    }


def save_predictions(path: Path, timestamps, y_true, y_pred) -> None:
    frame = pd.DataFrame({
        "delivery_timestamp": pd.to_datetime(timestamps).astype(str).to_numpy(),
        "actual_dart": np.asarray(y_true, dtype=float),
        "predicted_dart": np.asarray(y_pred, dtype=float),
    })
    frame["error"] = frame["predicted_dart"] - frame["actual_dart"]
    frame["absolute_error"] = frame["error"].abs()
    frame["model"] = MODEL_NAME
    frame.to_csv(path, index=False)


def monthly_performance(timestamps, y_true, y_pred) -> pd.DataFrame:
    frame = pd.DataFrame({
        "delivery_timestamp": pd.to_datetime(timestamps).to_numpy(),
        "actual_dart": np.asarray(y_true, dtype=float),
        "predicted_dart": np.asarray(y_pred, dtype=float),
    })
    frame["month"] = frame["delivery_timestamp"].dt.to_period("M").astype(str)
    rows = []
    for month, group in frame.groupby("month", sort=True):
        rows.append({
            "month": month,
            "n": int(len(group)),
            "mae": float(mean_absolute_error(group["actual_dart"], group["predicted_dart"])),
            "rmse": float(np.sqrt(mean_squared_error(group["actual_dart"], group["predicted_dart"]))),
            "bias": float((group["predicted_dart"] - group["actual_dart"]).mean()),
        })
    return pd.DataFrame(rows)


def fit_validation_model(X_train, y_train, X_val, y_val):
    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=35,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.20,
        reg_lambda=2.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    # LightGBM 4.7+: eval_X/eval_y replace deprecated eval_set.
    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric="l1",
        callbacks=[early_stopping(stopping_rounds=150, verbose=False), log_evaluation(0)],
    )
    best_iteration = int(model.best_iteration_ or 3000)
    return model, max(50, best_iteration)


def fit_final_model(X, y, iterations):
    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=int(iterations),
        learning_rate=0.02,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=35,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.20,
        reg_lambda=2.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X, y)
    return model


def update_comparison() -> None:
    """Refresh existing boosting metrics; verify test windows before ranking."""
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("lightgbm", "catboost", "xgboost"):
        metrics_path = ROOT / "results" / name / "test_metrics.csv"
        summary_path = ROOT / "results" / name / "run_summary.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = pd.read_csv(metrics_path)
            if metrics.empty:
                continue
            row = metrics.iloc[0].to_dict()
            row["model"] = name
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as file:
                    summary = json.load(file)
                row["test_start"] = summary.get("test_start_day", summary.get("test_start", ""))
                row["test_end"] = summary.get("test_end_day", summary.get("test_end", ""))
            rows.append(row)
        except Exception as exc:
            print(f"Warning: could not read {metrics_path}: {exc}")
    if rows:
        comparison = pd.DataFrame(rows)
        if "mae" in comparison.columns:
            comparison = comparison.sort_values("mae", ascending=True)
        comparison.to_csv(COMPARISON_DIR / "boosting_models_comparison.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM for Ontario DART forecasting")
    parser.add_argument("--input", type=str, default=None, help="Path to processed DART feature CSV")
    args = parser.parse_args()

    input_path = resolve_input(args.input)
    X, y, timestamps, _, feature_audit = load_data(input_path)
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        t_train, t_val, t_test,
        feature_names, split_info,
    ) = prepare_splits(X, y, timestamps)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    feature_audit.to_csv(RESULT_DIR / "feature_audit.csv", index=False)

    excluded_rt = feature_audit[
        feature_audit["reason"].eq("contemporaneous_realized_RT_price")
    ]["feature"].tolist()

    print("=" * 76)
    print("LIGHTGBM DART FORECAST — LEAKAGE-SCREENED FEATURES")
    print("=" * 76)
    print(f"Input: {input_path}")
    print(f"Observations: {len(y):,} | Predictors: {len(feature_names)}")
    print(f"Market days: {split_info['n_market_days']}")
    print(f"Train / validation / test: {len(y_train):,} / {len(y_val):,} / {len(y_test):,}")
    print(f"Test dates: {split_info['test_start_day']} to {split_info['test_end_day']}")
    print(f"Excluded contemporaneous RT price columns: {excluded_rt if excluded_rt else 'None detected'}")

    validation_model, best_iteration = fit_validation_model(X_train, y_train, X_val, y_val)
    validation_predictions = validation_model.predict(X_val)

    # Refit on train + validation; never use test outcomes for fitting.
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    final_model = fit_final_model(X_train_val, y_train_val, best_iteration)
    test_predictions = final_model.predict(X_test)

    # Training-only 95th percentile defines the spike threshold.
    spike_threshold = float(np.quantile(np.abs(y_train.to_numpy(dtype=float)), 0.95))
    validation_metrics = calculate_metrics(y_val, validation_predictions, spike_threshold)
    test_metrics = calculate_metrics(y_test, test_predictions, spike_threshold)

    save_predictions(RESULT_DIR / "validation_predictions.csv", t_val, y_val, validation_predictions)
    save_predictions(RESULT_DIR / "test_predictions.csv", t_test, y_test, test_predictions)
    pd.DataFrame([validation_metrics]).to_csv(RESULT_DIR / "validation_metrics.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(RESULT_DIR / "test_metrics.csv", index=False)
    monthly_performance(t_test, y_test, test_predictions).to_csv(
        RESULT_DIR / "monthly_test_performance.csv", index=False
    )
    joblib.dump(final_model, RESULT_DIR / "model.joblib")

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": np.asarray(final_model.feature_importances_, dtype=float),
    }).sort_values("importance", ascending=False)
    importance.to_csv(RESULT_DIR / "feature_importance.csv", index=False)

    summary = {
        "model": MODEL_NAME,
        "input_file": str(input_path),
        "observations": int(len(y)),
        "first_delivery_timestamp": str(timestamps.min()),
        "last_delivery_timestamp": str(timestamps.max()),
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_val)),
        "n_test": int(len(y_test)),
        "n_features": int(len(feature_names)),
        "features": feature_names,
        "best_iteration": int(best_iteration),
        "train_validation_end": str(t_val.iloc[-1]),
        "test_start": str(t_test.iloc[0]),
        "test_end": str(t_test.iloc[-1]),
        "split": "chronological by market day: 70% train / 15% validation / 15% test",
        "split_details": split_info,
        "excluded_contemporaneous_rt_price_columns": excluded_rt,
        "test_metrics": test_metrics,
        "important_warning": (
            "Name-based screening is not a complete leakage audit. Verify that every retained "
            "predictor is genuinely available at the day-ahead forecast origin."
        ),
    }
    with open(RESULT_DIR / "run_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=True)

    update_comparison()

    print("\nTEST METRICS")
    print(f"Best iteration: {best_iteration}")
    print(f"Test MAE:  {test_metrics['mae']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test R2:   {test_metrics['r2']:.4f}")
    print(f"Test Bias: {test_metrics['bias']:.4f}")
    print(f"Spike MAE: {test_metrics['spike_mae']:.4f}")
    print(f"Spike precision: {test_metrics['spike_precision']:.4f}")
    print(f"Spike recall: {test_metrics['spike_recall']:.4f}")
    print(f"Results: {RESULT_DIR}")
    print("=" * 76)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
