#!/usr/bin/env python3
"""LEAR-style hourly LASSO model for Ontario DART forecasting.

Key safeguards and corrections:
- Uses the same chronological market-day split as the other project models.
- Keeps every complete test market day; lag features use the latest available
  observed market day on or before D-2, D-7, and D-14 (never after that cutoff).
- Standardises X and y inside each cross-validation fold to improve LASSO
  numerical stability and avoid scale-dependent regularisation.
- Selects alpha with expanding-window TimeSeriesSplit using training data only.
- Refits each hourly model on train + validation before evaluating the test set.
- Excludes obvious realised/current DA and RT price fields from predictors.

Important: feature-name screening cannot prove that every remaining exogenous
feature was published by the D-2 forecast origin. Review feature_audit.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor


# ============================================================
# PROJECT PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "lear"
RESULT_DIR = ROOT / "results" / MODEL_NAME
COMPARISON_DIR = ROOT / "results" / "model_comparison"

DEFAULT_INPUTS = [
    ROOT / "datasets" / "processed" / "dart_historical_features.csv",
    ROOT / "datasets" / "processed" / "dart_training_features_provincial.csv",
    ROOT / "datasets" / "processed" / "dart_training_features.csv",
]

TARGET_ALIASES = ["target_dart", "dart", "dart_spread", "target"]
DATE_ALIASES = [
    "market_date", "delivery_date", "delivery_timestamp",
    "timestamp", "ts_utc", "datetime", "date",
]
HOUR_ALIASES = [
    "hour_ending", "hourending", "he", "hour", "hour_of_day",
    "delivery_hour", "hour_beginning",
]

# Conservative name-based screen for fields that are commonly realised only
# during/after delivery. Historical DART is added separately using D-2 lags.
LEAKAGE_TOKENS = [
    "dart",
    "rt_lmp", "rt_price", "rt_ozp", "real_time", "realtime",
    "actual_rt", "realized_rt", "realised_rt", "actual_real_time",
    "realized_real_time", "realised_real_time",
    "da_lmp", "da_price", "da_ozp", "day_ahead_price",
    "actual_da", "settlement", "realized_price", "realised_price",
    "actual_price", "price_actual", "spot_price",
    "lmp", "ozp", "price",
    "actual_load", "realized_load", "realised_load",
    "actual_demand", "actual_generation",
]

TRAIN_SHARE = 0.70
VALIDATION_SHARE = 0.15
LAG_DAYS = [2, 7, 14]
RANDOM_SEED = 42

# The target is standardised within each CV fit, so these alpha values are on
# a comparable scale across delivery hours.
ALPHAS = np.logspace(-3, 0.5, 25)
LASSO_MAX_ITER = 200_000
LASSO_TOL = 1e-4
CV_SPLITS = 4
CV_GAP = 2


# ============================================================
# INPUT HELPERS
# ============================================================

def find_col(columns: list[str], aliases: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): str(c) for c in columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


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


def make_timestamps(
    raw: pd.DataFrame,
    date_col: str,
    hour_col: str | None,
) -> pd.Series:
    ts = pd.to_datetime(raw[date_col], errors="coerce")

    # If the date column contains midnight dates and a separate hour-ending
    # column is supplied, convert HE 1..24 to hour-beginning 00..23.
    if (
        hour_col is not None
        and ts.notna().any()
        and ts.dt.hour.fillna(0).eq(0).all()
    ):
        h = pd.to_numeric(raw[hour_col], errors="coerce")
        if h.notna().all() and h.between(1, 24).all():
            ts = ts + pd.to_timedelta(h - 1, unit="h")
        elif h.notna().all() and h.between(0, 23).all():
            ts = ts + pd.to_timedelta(h, unit="h")

    return ts


def make_calendar_features(dates: pd.Series) -> pd.DataFrame:
    dow = dates.dt.day_of_week.astype(float)
    month = dates.dt.month.astype(float)
    doy = dates.dt.day_of_year.astype(float)

    return pd.DataFrame({
        "calendar_dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "calendar_dow_cos": np.cos(2 * np.pi * dow / 7.0),
        "calendar_month_sin": np.sin(2 * np.pi * (month - 1) / 12.0),
        "calendar_month_cos": np.cos(2 * np.pi * (month - 1) / 12.0),
        "calendar_year_sin": np.sin(2 * np.pi * (doy - 1) / 365.25),
        "calendar_year_cos": np.cos(2 * np.pi * (doy - 1) / 365.25),
    })


def add_asof_dart_lags(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    hours: pd.Series,
) -> pd.DataFrame:
    """Add same-hour DART lags using only observations available by D-lag.

    For a delivery date D, each lag selects the latest observed complete
    market date <= D-lag calendar days. This preserves the D-2 information
    cutoff even when an exact calendar lag date is absent from the dataset.
    """
    unique_dates = pd.DatetimeIndex(
        pd.Series(dates).drop_duplicates().sort_values()
    )

    target_lookup = pd.Series(
        y.to_numpy(dtype=float),
        index=pd.MultiIndex.from_arrays([
            dates.to_numpy(),
            hours.to_numpy(dtype=int),
        ]),
    )

    out = X.copy()

    for lag in LAG_DAYS:
        lag_dates: list[Any] = []

        for delivery_date in dates:
            cutoff = delivery_date - pd.Timedelta(days=lag)
            location = unique_dates.searchsorted(cutoff, side="right") - 1
            if location < 0:
                lag_dates.append(pd.NaT)
            else:
                lag_dates.append(unique_dates[location])

        lag_index = pd.MultiIndex.from_arrays([
            np.asarray(lag_dates),
            hours.to_numpy(dtype=int),
        ])

        column = f"dart_lag_{lag}d_same_hour"
        out[column] = target_lookup.reindex(lag_index).to_numpy(dtype=float)

    return out


def load_data(path: Path):
    raw = pd.read_csv(path, low_memory=False)
    if raw.empty:
        raise ValueError(f"Input CSV is empty: {path}")

    target_col = find_col(list(raw.columns), TARGET_ALIASES)
    date_col = find_col(list(raw.columns), DATE_ALIASES)
    hour_col = find_col(list(raw.columns), HOUR_ALIASES)

    if target_col is None:
        raise ValueError(f"DART target not found. Available columns: {list(raw.columns)}")
    if date_col is None:
        raise ValueError(f"Delivery date/timestamp not found. Available columns: {list(raw.columns)}")

    ts = make_timestamps(raw, date_col, hour_col)
    y = pd.to_numeric(raw[target_col], errors="coerce")
    valid = ts.notna() & y.notna() & np.isfinite(y)
    raw = raw.loc[valid].copy()
    ts = ts.loc[valid]
    y = y.loc[valid]

    order = ts.sort_values(kind="stable").index
    raw = raw.loc[order].reset_index(drop=True)
    ts = ts.loc[order].reset_index(drop=True)
    y = y.loc[order].reset_index(drop=True).astype(float)

    if ts.duplicated().any():
        raise ValueError(
            f"Found {int(ts.duplicated().sum())} duplicate delivery timestamps. "
            "The processed Ontario file should contain one row per delivery hour. "
            "Aggregate/filter zones before fitting."
        )

    dates = ts.dt.normalize()
    hours = ts.dt.hour.astype(int)

    # Keep complete 24-hour dates, matching the project's complete-market-day
    # sample and ensuring the model's output can be compared hour by hour.
    day_summary = pd.DataFrame({"date": dates, "hour": hours}).groupby("date").agg(
        n_rows=("hour", "size"),
        n_hours=("hour", "nunique"),
    )
    complete_dates = day_summary.index[
        (day_summary["n_rows"] == 24) & (day_summary["n_hours"] == 24)
    ]
    incomplete_dates = day_summary.index.difference(complete_dates)

    if len(complete_dates) == 0:
        raise ValueError("No complete 24-hour market days found in the input.")

    keep_days = dates.isin(complete_dates)
    raw = raw.loc[keep_days].reset_index(drop=True)
    ts = ts.loc[keep_days].reset_index(drop=True)
    y = y.loc[keep_days].reset_index(drop=True)
    dates = ts.dt.normalize()
    hours = ts.dt.hour.astype(int)

    X = pd.DataFrame(index=raw.index)
    audit: list[dict[str, Any]] = []
    identifiers = {target_col.lower(), date_col.lower()}
    if hour_col:
        identifiers.add(hour_col.lower())

    for col in raw.columns:
        name = str(col)
        low = name.strip().lower()

        if low in identifiers:
            audit.append({
                "feature": name,
                "included": False,
                "reason": "Target/date/hour identifier.",
            })
            continue

        matched = [token for token in LEAKAGE_TOKENS if token in low]
        if matched:
            audit.append({
                "feature": name,
                "included": False,
                "reason": "Excluded by realised-price/target name screen: " + ", ".join(matched),
            })
            continue

        numeric = pd.to_numeric(raw[col], errors="coerce")
        observed = int(raw[col].notna().sum())
        numeric_share = numeric.notna().sum() / observed if observed else 0.0

        if observed == 0 or numeric_share < 0.90:
            audit.append({
                "feature": name,
                "included": False,
                "reason": "Not sufficiently numeric or has no observed values.",
            })
            continue

        X[name] = numeric.replace([np.inf, -np.inf], np.nan)
        audit.append({
            "feature": name,
            "included": True,
            "reason": "Numeric candidate retained; verify the data vintage was available by D-2.",
        })

    calendar = make_calendar_features(dates)
    for name in calendar.columns:
        X[name] = calendar[name].to_numpy(dtype=float)
        audit.append({
            "feature": name,
            "included": True,
            "reason": "Engineered from delivery calendar; known in advance.",
        })

    X = add_asof_dart_lags(X, y, dates, hours)
    for lag in LAG_DAYS:
        audit.append({
            "feature": f"dart_lag_{lag}d_same_hour",
            "included": True,
            "reason": f"Same-hour DART from latest observed market date on or before D-{lag}; never later than cutoff.",
        })

    X = X.replace([np.inf, -np.inf], np.nan)

    metadata = {
        "target_column": target_col,
        "date_column": date_col,
        "hour_column": hour_col,
        "n_incomplete_dates_removed": int(len(incomplete_dates)),
        "incomplete_dates_removed": [str(d.date()) for d in incomplete_dates],
    }

    return X, y, ts, dates, hours, pd.DataFrame(audit), metadata


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def split_dates(dates: pd.Series):
    unique = pd.DatetimeIndex(pd.Series(dates).drop_duplicates().sort_values())
    n = len(unique)
    train_end = int(n * TRAIN_SHARE)
    validation_end = int(n * (TRAIN_SHARE + VALIDATION_SHARE))

    if train_end < 100 or validation_end <= train_end or n - validation_end < 30:
        raise ValueError(f"Insufficient complete market days for split: {n}")

    return unique[:train_end], unique[train_end:validation_end], unique[validation_end:]


# ============================================================
# MODEL PIPELINE AND HYPERPARAMETER SELECTION
# ============================================================

def make_pipeline(alpha: float = 1.0) -> TransformedTargetRegressor:
    """Scale predictors and target within each fit/fold before LASSO."""
    x_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lasso", Lasso(
            alpha=alpha,
            max_iter=LASSO_MAX_ITER,
            tol=LASSO_TOL,
            selection="cyclic",
            random_state=RANDOM_SEED,
        )),
    ])

    return TransformedTargetRegressor(
        regressor=x_pipeline,
        transformer=StandardScaler(),
    )


def fit_hour(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_train_validation: pd.DataFrame,
    y_train_validation: pd.Series,
):
    """Tune alpha on training only, then refit for validation/test use."""
    missing_rate = X_train.isna().mean()
    columns = [
        c for c in X_train.columns
        if missing_rate[c] <= 0.40
        and X_train[c].notna().any()
        and X_train[c].nunique(dropna=True) > 1
    ]

    if not columns:
        raise ValueError("No usable predictors for one delivery hour.")

    # Per-hour sample size is about 350 observations. Four expanding windows
    # with a 2-observation gap mimic the D-2 forecast information boundary.
    cv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=CV_GAP)
    search = GridSearchCV(
        estimator=make_pipeline(),
        param_grid={"regressor__lasso__alpha": ALPHAS},
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
        return_train_score=False,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        search.fit(X_train[columns], y_train)

    convergence_warnings = sum(
        issubclass(item.category, ConvergenceWarning)
        for item in caught
    )

    alpha = float(search.best_params_["regressor__lasso__alpha"])
    validation_model = search.best_estimator_

    final_model = make_pipeline(alpha)
    final_model.fit(
        X_train_validation[columns],
        y_train_validation,
    )

    return (
        validation_model,
        final_model,
        alpha,
        columns,
        float(-search.best_score_),
        int(convergence_warnings),
    )


# ============================================================
# METRICS AND OUTPUTS
# ============================================================

def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    spike_threshold: float,
) -> dict[str, Any]:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    abs_error = np.abs(actual - pred)
    spike = np.abs(actual) >= spike_threshold
    predicted_spike = np.abs(pred) >= spike_threshold

    spike_precision = (
        float(np.sum(spike & predicted_spike) / np.sum(predicted_spike))
        if predicted_spike.any() else np.nan
    )
    spike_recall = (
        float(np.sum(spike & predicted_spike) / np.sum(spike))
        if spike.any() else np.nan
    )

    return {
        "model": MODEL_NAME,
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "r2": float(r2_score(actual, pred)),
        "bias": float(np.mean(pred - actual)),
        "direction_accuracy": float(np.mean(np.sign(actual) == np.sign(pred))),
        "spike_threshold_abs_dart": float(spike_threshold),
        "spike_n": int(spike.sum()),
        "spike_mae": float(abs_error[spike].mean()) if spike.any() else np.nan,
        "spike_recall": spike_recall,
        "spike_precision": spike_precision,
    }


def make_predictions(
    ts: pd.Series,
    hours: pd.Series,
    actual: pd.Series,
    predicted: np.ndarray,
) -> pd.DataFrame:
    out = pd.DataFrame({
        "delivery_timestamp": pd.to_datetime(ts).dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "delivery_hour": np.asarray(hours, dtype=int),
        "actual_dart": np.asarray(actual, dtype=float),
        "predicted_dart": np.asarray(predicted, dtype=float),
    })
    out["error"] = out["predicted_dart"] - out["actual_dart"]
    out["absolute_error"] = out["error"].abs()
    out["model"] = MODEL_NAME
    return out.sort_values("delivery_timestamp", kind="stable").reset_index(drop=True)


def monthly_results(pred: pd.DataFrame) -> pd.DataFrame:
    temp = pred.copy()
    temp["delivery_timestamp"] = pd.to_datetime(temp["delivery_timestamp"])
    temp["month"] = temp["delivery_timestamp"].dt.to_period("M").astype(str)
    rows = []

    for month, group in temp.groupby("month", sort=True):
        rows.append({
            "month": month,
            "n": int(len(group)),
            "mae": float(mean_absolute_error(group["actual_dart"], group["predicted_dart"])),
            "rmse": float(np.sqrt(mean_squared_error(group["actual_dart"], group["predicted_dart"]))),
            "bias": float((group["predicted_dart"] - group["actual_dart"]).mean()),
        })

    return pd.DataFrame(rows)


def update_comparison() -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    names = [
        "hour_of_week_median", "gaussian_gpd", "lightgbm",
        "catboost", "xgboost", "lear",
    ]
    rows = []

    for name in names:
        folder = ROOT / "results" / name
        metrics_path = folder / "test_metrics.csv"
        summary_path = folder / "run_summary.json"
        if not metrics_path.is_file():
            continue

        try:
            data = pd.read_csv(metrics_path)
            if data.empty:
                continue
            row = {str(k).lower(): v for k, v in data.iloc[0].to_dict().items()}
            test_start = None
            test_end = None
            if summary_path.is_file():
                with open(summary_path, "r", encoding="utf-8") as file:
                    summary = json.load(file)
                test_start = summary.get("test_start")
                test_end = summary.get("test_end")
            rows.append({
                "model": name,
                "n": row.get("n", row.get("observations", np.nan)),
                "test_start": test_start,
                "test_end": test_end,
                "mae": row.get("mae", np.nan),
                "rmse": row.get("rmse", np.nan),
                "bias": row.get("bias", np.nan),
                "spike_mae": row.get("spike_mae", np.nan),
                "spike_precision": row.get("spike_precision", np.nan),
                "spike_recall": row.get("spike_recall", np.nan),
            })
        except Exception as exc:
            print(f"Warning: could not read {metrics_path}: {exc}")

    if rows:
        comparison = pd.DataFrame(rows)
        comparison = comparison.sort_values("mae", na_position="last")
        comparison.to_csv(
            COMPARISON_DIR / "point_forecast_model_comparison.csv",
            index=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 24 hour-specific LEAR-style LASSO DART models"
    )
    parser.add_argument("--input", default=None, help="Processed DART feature CSV")
    args = parser.parse_args()

    input_path = resolve_input(args.input)
    X, y, ts, dates, hours, audit, metadata = load_data(input_path)

    train_dates, val_dates, test_dates = split_dates(dates)
    train_mask = dates.isin(train_dates)
    val_mask = dates.isin(val_dates)
    test_mask = dates.isin(test_dates)

    lag_cols = [f"dart_lag_{lag}d_same_hour" for lag in LAG_DAYS]
    lag_eligible = X[lag_cols].notna().all(axis=1)

    # Keep the original date windows intact. Only the initial training rows
    # without sufficient pre-sample D-14 history are removed. Validation and
    # test should remain complete 24-hour market days.
    train_mask &= lag_eligible
    val_mask &= lag_eligible
    test_mask &= lag_eligible

    if val_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("No validation/test rows remain after lag construction.")

    day_counts = pd.DataFrame({
        "date": dates.loc[test_mask],
        "hour": hours.loc[test_mask],
    }).groupby("date")["hour"].nunique()
    if len(day_counts) != len(test_dates) or not day_counts.eq(24).all():
        raise ValueError(
            "The test set is not 24 observations for every planned test date. "
            "Inspect missing target hours and lag construction before comparing models."
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(RESULT_DIR / "feature_audit.csv", index=False)

    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    fitted_models: dict[int, dict[str, Any]] = {}
    alpha_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    total_convergence_warnings = 0

    for hour in sorted(hours.unique()):
        hour_mask = hours.eq(hour)
        tr = train_mask & hour_mask
        va = val_mask & hour_mask
        te = test_mask & hour_mask

        Xtr = X.loc[tr].reset_index(drop=True)
        ytr = y.loc[tr].reset_index(drop=True)
        Xva = X.loc[va].reset_index(drop=True)
        yva = y.loc[va].reset_index(drop=True)
        Xte = X.loc[te].reset_index(drop=True)
        yte = y.loc[te].reset_index(drop=True)

        tva = ts.loc[va].reset_index(drop=True)
        tte = ts.loc[te].reset_index(drop=True)
        hva = hours.loc[va].reset_index(drop=True)
        hte = hours.loc[te].reset_index(drop=True)

        if min(len(Xtr), len(Xva), len(Xte)) < 10:
            raise ValueError(f"Insufficient observations for delivery hour {hour}.")

        Xtv = pd.concat([Xtr, Xva], ignore_index=True)
        ytv = pd.concat([ytr, yva], ignore_index=True)

        val_model, final_model, alpha, columns, cv_mae, warning_count = fit_hour(
            Xtr, ytr, Xtv, ytv
        )
        total_convergence_warnings += warning_count

        val_hat = val_model.predict(Xva[columns])
        test_hat = final_model.predict(Xte[columns])

        validation_parts.append(make_predictions(tva, hva, yva, val_hat))
        test_parts.append(make_predictions(tte, hte, yte, test_hat))

        fitted_models[int(hour)] = {
            "pipeline": final_model,
            "features": columns,
            "alpha": alpha,
        }

        alpha_rows.append({
            "delivery_hour": int(hour),
            "selected_alpha_standardized_target": alpha,
            "training_cv_mae": cv_mae,
            "n_train": int(len(Xtr)),
            "n_validation": int(len(Xva)),
            "n_test": int(len(Xte)),
            "n_features": int(len(columns)),
            "convergence_warnings_during_cv": int(warning_count),
        })

        lasso = final_model.regressor_.named_steps["lasso"]
        for feature, coef in zip(columns, lasso.coef_):
            coefficient_rows.append({
                "delivery_hour": int(hour),
                "feature": feature,
                "coefficient_standardized": float(coef),
                "absolute_coefficient": float(abs(coef)),
            })

        print(
            f"Hour {int(hour):02d}: train={len(Xtr):,}, "
            f"validation={len(Xva):,}, test={len(Xte):,}, "
            f"alpha={alpha:.6g}, CV convergence warnings={warning_count}"
        )

    validation_pred = pd.concat(validation_parts, ignore_index=True)
    test_pred = pd.concat(test_parts, ignore_index=True)
    validation_pred = validation_pred.sort_values("delivery_timestamp").reset_index(drop=True)
    test_pred = test_pred.sort_values("delivery_timestamp").reset_index(drop=True)

    # Training-only 95th percentile absolute-DART threshold.
    spike_threshold = float(
        np.quantile(np.abs(y.loc[train_mask].to_numpy(dtype=float)), 0.95)
    )

    validation_metrics = calculate_metrics(
        validation_pred["actual_dart"],
        validation_pred["predicted_dart"].to_numpy(),
        spike_threshold,
    )
    test_metrics = calculate_metrics(
        test_pred["actual_dart"],
        test_pred["predicted_dart"].to_numpy(),
        spike_threshold,
    )

    validation_pred.to_csv(RESULT_DIR / "validation_predictions.csv", index=False)
    test_pred.to_csv(RESULT_DIR / "test_predictions.csv", index=False)
    pd.DataFrame([validation_metrics]).to_csv(
        RESULT_DIR / "validation_metrics.csv", index=False
    )
    pd.DataFrame([test_metrics]).to_csv(
        RESULT_DIR / "test_metrics.csv", index=False
    )
    monthly_results(test_pred).to_csv(
        RESULT_DIR / "monthly_test_performance.csv", index=False
    )
    pd.DataFrame(alpha_rows).to_csv(
        RESULT_DIR / "selected_alpha_by_hour.csv", index=False
    )

    coefficient_frame = pd.DataFrame(coefficient_rows)
    if not coefficient_frame.empty:
        importance = coefficient_frame.groupby("feature", as_index=False).agg(
            mean_absolute_coefficient=("absolute_coefficient", "mean"),
            median_absolute_coefficient=("absolute_coefficient", "median"),
            hours_selected=(
                "absolute_coefficient",
                lambda values: int((values > 1e-10).sum()),
            ),
        ).sort_values("mean_absolute_coefficient", ascending=False)
    else:
        importance = pd.DataFrame(columns=[
            "feature", "mean_absolute_coefficient",
            "median_absolute_coefficient", "hours_selected",
        ])
    importance.to_csv(RESULT_DIR / "feature_importance.csv", index=False)

    joblib.dump({
        "model": MODEL_NAME,
        "models_by_hour": fitted_models,
        "lag_days": LAG_DAYS,
        "lag_policy": "latest observed complete market date <= delivery date minus lag days",
        "forecast_cutoff": "D-2",
    }, RESULT_DIR / "model.joblib")

    summary = {
        "model": MODEL_NAME,
        "description": "24 hour-specific LASSO regressions with fold-wise X/y standardisation",
        "input_file": str(input_path),
        "target_column": metadata["target_column"],
        "date_column": metadata["date_column"],
        "hour_column": metadata["hour_column"],
        "observations_raw_complete_days": int(len(y)),
        "market_days_raw": int(dates.nunique()),
        "incomplete_dates_removed": metadata["incomplete_dates_removed"],
        "first_delivery_timestamp": str(ts.min()),
        "last_delivery_timestamp": str(ts.max()),
        "train_start": str(train_dates.min()),
        "train_end": str(train_dates.max()),
        "validation_start": str(val_dates.min()),
        "validation_end": str(val_dates.max()),
        "test_start": str(test_dates.min()),
        "test_end": str(test_dates.max()),
        "train_market_days_before_lag_filter": int(len(train_dates)),
        "validation_market_days_before_lag_filter": int(len(val_dates)),
        "test_market_days_before_lag_filter": int(len(test_dates)),
        "n_train_after_lags": int(train_mask.sum()),
        "n_validation_after_lags": int(val_mask.sum()),
        "n_test_after_lags": int(test_mask.sum()),
        "n_hour_models": int(len(fitted_models)),
        "lag_days": LAG_DAYS,
        "lag_policy": "latest complete observed market date on or before D-2/D-7/D-14",
        "cv": f"{CV_SPLITS}-fold expanding TimeSeriesSplit, gap={CV_GAP}, training-only MAE alpha selection",
        "alpha_grid": ALPHAS.tolist(),
        "lasso_max_iter": LASSO_MAX_ITER,
        "lasso_tolerance": LASSO_TOL,
        "target_standardisation": "StandardScaler fitted separately inside each CV training fold",
        "convergence_warnings_during_cv": int(total_convergence_warnings),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "feature_audit": str(RESULT_DIR / "feature_audit.csv"),
        "warning": "Verify every retained exogenous feature's publication/vintage time before treating the results as an ex-ante D-2 forecast.",
    }

    with open(RESULT_DIR / "run_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=True)

    update_comparison()

    print("=" * 76)
    print("ONTARIO DART: LEAR-STYLE HOURLY LASSO — CORRECTED")
    print("=" * 76)
    print(f"Input: {input_path}")
    print(f"Complete market days: {dates.nunique():,}")
    print(f"Raw hourly observations: {len(y):,}")
    print(
        f"Date split (before lag filter): train={len(train_dates):,}, "
        f"validation={len(val_dates):,}, test={len(test_dates):,} market days"
    )
    print(
        f"Train / validation / test after lag filter: "
        f"{int(train_mask.sum()):,} / {int(val_mask.sum()):,} / {int(test_mask.sum()):,}"
    )
    print(f"Test dates: {test_dates.min().date()} to {test_dates.max().date()}")
    print(f"Excluded incomplete market dates: {metadata['n_incomplete_dates_removed']}")
    print(f"CV convergence warnings: {total_convergence_warnings}")
    print("\nTEST METRICS")
    print(f"MAE:  {test_metrics['mae']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"R2:   {test_metrics['r2']:.4f}")
    print(f"Bias: {test_metrics['bias']:.4f}")
    print(f"Spike MAE: {test_metrics['spike_mae']:.4f}")
    print(f"Spike precision: {test_metrics['spike_precision']}")
    print(f"Spike recall: {test_metrics['spike_recall']}")
    print(f"Results: {RESULT_DIR}")
    print("Review feature_audit.csv for D-2 availability of retained exogenous predictors.")
    print("=" * 76)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
