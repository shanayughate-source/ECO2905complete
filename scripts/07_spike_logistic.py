#!/usr/bin/env python3
"""Two-sided DART spike logistic model for Ontario electricity forecasting.

Fits separate regularised logistic regressions for negative and positive DART
spikes. Spike cutoffs are calibrated from training data, probability cutoffs
are selected on validation data, and final probabilities are evaluated on an
untouched chronological test period.

Input: datasets/processed/dart_historical_features.csv by default.
Output: results/dart_spike_logistic/.

This is an Ontario-adapted spike-classification model inspired by the
positive/negative logistic approach in Hubert, Lolas & Sircar (2026). It is not
an exact replication of their NYISO multi-zone trading implementation.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit


# ============================================================
# PROJECT SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "dart_spike_logistic"
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

# Only allow price/DART-like columns when their names explicitly indicate
# a historical lag, rolling statistic, or past observation. This is still a
# name-based screen, so feature_audit.csv must be reviewed manually.
BLOCK_TOKENS = [
    "target_dart", "actual_dart", "realized_dart", "realised_dart",
    "dart_spread", "rt_lmp", "rt_price", "rt_ozp", "real_time",
    "realtime", "actual_rt", "realized_rt", "realised_rt",
    "settlement", "actual_price", "realized_price", "realised_price",
    "spot_price", "da_lmp", "da_price", "da_ozp", "day_ahead_price",
    "actual_da", "price_actual",
]

HISTORICAL_MARKERS = [
    "lag", "rolling", "historical", "history", "past", "prior",
    "previous", "_d_2", "_d2", "_d_7", "_d7", "_d_14", "_d14",
    "24h", "48h", "72h", "168h", "336h",
]

TRAIN_SHARE = 0.70
VALIDATION_SHARE = 0.15
SPIKE_QUANTILE = 0.95
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
CV_SPLITS = 4
CV_GAP_DAYS = 2
MAX_ITER = 10000
TOL = 1e-5
RANDOM_STATE = 42


# ============================================================
# INPUT AND COLUMN HELPERS
# ============================================================

def find_column(columns: list[str], aliases: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): str(c) for c in columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def resolve_input(input_arg: str | None) -> Path:
    if input_arg:
        path = Path(input_arg).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Input CSV not found: {path}")
        return path
    for path in DEFAULT_INPUTS:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not find a processed DART feature CSV. Checked:\n"
        + "\n".join(f"  - {p}" for p in DEFAULT_INPUTS)
        + "\nPass the correct file with --input."
    )


def make_timestamps(raw: pd.DataFrame, date_col: str, hour_col: str | None) -> pd.Series:
    timestamps = pd.to_datetime(raw[date_col], errors="coerce")
    if hour_col is not None and timestamps.notna().any():
        hours = pd.to_numeric(raw[hour_col], errors="coerce")
        if timestamps.dt.hour.fillna(0).eq(0).all():
            if hours.notna().all() and hours.between(1, 24).all():
                timestamps = timestamps + pd.to_timedelta(hours - 1, unit="h")
            elif hours.notna().all() and hours.between(0, 23).all():
                timestamps = timestamps + pd.to_timedelta(hours, unit="h")
    return timestamps


def is_historical_name(name: str) -> bool:
    low = name.lower()
    return any(marker in low for marker in HISTORICAL_MARKERS)


def feature_exclusion_reason(name: str, target_col: str, date_col: str,
                             hour_col: str | None) -> str | None:
    low = name.lower().strip()
    if low in {target_col.lower(), date_col.lower()}:
        return "Target/date identifier."
    if hour_col and low == hour_col.lower():
        return "Hour identifier; calendar encodings are added separately."

    matches = [token for token in BLOCK_TOKENS if token in low]
    if matches and not is_historical_name(low):
        return "Potential contemporaneous realised-price/target field: " + ", ".join(matches)
    if matches and is_historical_name(low):
        return None
    return None


# ============================================================
# DATA LOADING, CALENDAR FEATURES, AND D-2 LAGS
# ============================================================

def add_calendar_features(X: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
    out = X.copy()
    hour = timestamps.dt.hour.astype(float)
    dow = timestamps.dt.day_of_week.astype(float)
    month = timestamps.dt.month.astype(float)
    day_of_year = timestamps.dt.day_of_year.astype(float)

    out["calendar_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["calendar_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["calendar_dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["calendar_dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["calendar_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    out["calendar_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    out["calendar_year_sin"] = np.sin(2 * np.pi * (day_of_year - 1) / 365.25)
    out["calendar_year_cos"] = np.cos(2 * np.pi * (day_of_year - 1) / 365.25)
    return out


def add_historical_dart_lags(X: pd.DataFrame, y: pd.Series,
                             dates: pd.Series, hours: pd.Series,
                             lag_days: tuple[int, ...] = (2, 7, 14)) -> pd.DataFrame:
    """Add same-hour DART lags using only dates <= delivery date - lag."""
    out = X.copy()
    unique_dates = pd.DatetimeIndex(pd.Series(dates).drop_duplicates().sort_values())
    target_lookup = pd.Series(
        y.to_numpy(dtype=float),
        index=pd.MultiIndex.from_arrays([
            dates.to_numpy(), hours.to_numpy(dtype=int)
        ]),
    )
    for lag in lag_days:
        lag_dates = []
        for delivery_date in dates:
            cutoff = delivery_date - pd.Timedelta(days=lag)
            pos = unique_dates.searchsorted(cutoff, side="right") - 1
            lag_dates.append(pd.NaT if pos < 0 else unique_dates[pos])
        lag_index = pd.MultiIndex.from_arrays([
            np.asarray(lag_dates), hours.to_numpy(dtype=int)
        ])
        out[f"dart_lag_{lag}d_same_hour"] = target_lookup.reindex(lag_index).to_numpy(dtype=float)
    return out


def load_data(path: Path):
    raw = pd.read_csv(path, low_memory=False)
    if raw.empty:
        raise ValueError(f"Input CSV is empty: {path}")

    target_col = find_column(list(raw.columns), TARGET_ALIASES)
    date_col = find_column(list(raw.columns), DATE_ALIASES)
    hour_col = find_column(list(raw.columns), HOUR_ALIASES)
    if target_col is None:
        raise ValueError(f"No DART target found. Columns: {list(raw.columns)}")
    if date_col is None:
        raise ValueError(f"No delivery-date column found. Columns: {list(raw.columns)}")

    timestamps = make_timestamps(raw, date_col, hour_col)
    target = pd.to_numeric(raw[target_col], errors="coerce")
    valid = timestamps.notna() & target.notna() & np.isfinite(target)
    raw = raw.loc[valid].copy()
    timestamps = timestamps.loc[valid]
    target = target.loc[valid]

    order = timestamps.sort_values(kind="stable").index
    raw = raw.loc[order].reset_index(drop=True)
    timestamps = timestamps.loc[order].reset_index(drop=True)
    y = target.loc[order].reset_index(drop=True).astype(float)

    if timestamps.duplicated().any():
        dupes = int(timestamps.duplicated().sum())
        raise ValueError(
            f"Found {dupes} duplicate delivery timestamps. "
            "Filter to one Ontario price series/zone before modelling."
        )

    dates = timestamps.dt.normalize()
    hours = timestamps.dt.hour.astype(int)

    # Retain complete 24-hour dates only, matching the project's market-day panel.
    day_summary = pd.DataFrame({"date": dates, "hour": hours}).groupby("date").agg(
        n_rows=("hour", "size"), n_hours=("hour", "nunique")
    )
    complete_dates = day_summary.index[
        (day_summary["n_rows"] == 24) & (day_summary["n_hours"] == 24)
    ]
    removed_dates = day_summary.index.difference(complete_dates)
    keep = dates.isin(complete_dates)
    raw = raw.loc[keep].reset_index(drop=True)
    timestamps = timestamps.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)
    dates = timestamps.dt.normalize()
    hours = timestamps.dt.hour.astype(int)

    X = pd.DataFrame(index=raw.index)
    audit_rows = []
    for col in raw.columns:
        name = str(col)
        reason = feature_exclusion_reason(name, target_col, date_col, hour_col)
        if reason:
            audit_rows.append({"feature": name, "included": False, "reason": reason})
            continue
        numeric = pd.to_numeric(raw[col], errors="coerce")
        observed = int(raw[col].notna().sum())
        numeric_share = numeric.notna().sum() / observed if observed else 0.0
        if observed == 0 or numeric_share < 0.90:
            audit_rows.append({
                "feature": name, "included": False,
                "reason": "Not sufficiently numeric."
            })
            continue
        X[name] = numeric.replace([np.inf, -np.inf], np.nan)
        audit_rows.append({
            "feature": name, "included": True,
            "reason": "Numeric candidate; verify publication/vintage time at D-2."
        })

    X = add_calendar_features(X, timestamps)
    for col in X.columns:
        if col not in {r["feature"] for r in audit_rows}:
            audit_rows.append({
                "feature": col, "included": True,
                "reason": "Engineered calendar feature or historical DART lag."
            })

    X = add_historical_dart_lags(X, y, dates, hours)
    X = X.replace([np.inf, -np.inf], np.nan)
    for lag in (2, 7, 14):
        audit_rows.append({
            "feature": f"dart_lag_{lag}d_same_hour",
            "included": True,
            "reason": f"Same-hour DART from latest market date <= D-{lag}."
        })

    metadata = {
        "target_column": target_col,
        "date_column": date_col,
        "hour_column": hour_col,
        "incomplete_dates_removed": [str(d.date()) for d in removed_dates],
    }
    return X, y, timestamps, dates, hours, pd.DataFrame(audit_rows), metadata


# ============================================================
# DATE SPLITS AND MODEL PIPELINE
# ============================================================

def chronological_dates(dates):
    unique = pd.DatetimeIndex(pd.Series(dates).drop_duplicates().sort_values())
    n = len(unique)
    train_end = int(n * TRAIN_SHARE)
    val_end = int(n * (TRAIN_SHARE + VALIDATION_SHARE))
    if train_end < 100 or val_end <= train_end or n - val_end < 30:
        raise ValueError(f"Insufficient complete market days: {n}")
    return unique[:train_end], unique[train_end:val_end], unique[val_end:]


def make_pipeline(C: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=False)),
        ("scaler", RobustScaler()),
        ("logistic", LogisticRegression(
            C=float(C), penalty="l2", solver="lbfgs",
            max_iter=MAX_ITER, tol=1e-6,
            class_weight=None, random_state=RANDOM_STATE,
        )),
    ])


def tune_logistic_C(X: pd.DataFrame, y_binary: pd.Series,
                    dates: pd.Series, columns: list[str],
                    label_name: str):
    """Select C using expanding chronological folds over market dates."""
    unique_dates = pd.DatetimeIndex(pd.Series(dates).drop_duplicates().sort_values())
    splitter = TimeSeriesSplit(n_splits=CV_SPLITS, gap=CV_GAP_DAYS)
    date_splits = list(splitter.split(unique_dates))
    score_rows = []

    for C in C_GRID:
        fold_losses = []
        for train_idx, val_idx in date_splits:
            fold_train_dates = unique_dates[train_idx]
            fold_val_dates = unique_dates[val_idx]
            fold_train_mask = dates.isin(fold_train_dates).to_numpy()
            fold_val_mask = dates.isin(fold_val_dates).to_numpy()
            y_fold_train = y_binary.iloc[np.flatnonzero(fold_train_mask)]
            y_fold_val = y_binary.iloc[np.flatnonzero(fold_val_mask)]
            if y_fold_train.nunique() < 2 or len(y_fold_val) == 0:
                continue
            model = make_pipeline(C)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(X.loc[fold_train_mask, columns], y_fold_train)
            p = model.predict_proba(X.loc[fold_val_mask, columns])[:, 1]
            p = np.clip(p, 1e-7, 1 - 1e-7)
            loss = log_loss(y_fold_val, p, labels=[0, 1])
            fold_losses.append(float(loss))
        mean_loss = float(np.mean(fold_losses)) if fold_losses else np.inf
        score_rows.append({"side": label_name, "C": float(C), "mean_cv_log_loss": mean_loss,
                           "valid_folds": len(fold_losses)})

    scores = pd.DataFrame(score_rows)
    valid_scores = scores[np.isfinite(scores["mean_cv_log_loss"])]
    if valid_scores.empty:
        raise ValueError(
            f"Could not tune logistic regression for {label_name}; "
            "one or more chronological folds may contain only one class."
        )
    best_C = float(valid_scores.sort_values("mean_cv_log_loss").iloc[0]["C"])
    return best_C, scores


# ============================================================
# VALIDATION PROBABILITY THRESHOLD
# ============================================================

def choose_probability_cutoff(y_true, probabilities, beta: float = 1.0):
    """Choose F-beta cutoff on validation only; no test-set tuning."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.concatenate([
        np.linspace(0.01, 0.99, 99),
        np.quantile(p, np.linspace(0.05, 0.99, 50)),
    ]))
    best = None
    beta2 = beta ** 2
    for cutoff in candidates:
        pred = p >= cutoff
        tp = int(np.sum((y == 1) & pred))
        fp = int(np.sum((y == 0) & pred))
        fn = int(np.sum((y == 1) & (~pred)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        denom = beta2 * precision + recall
        score = (1 + beta2) * precision * recall / denom if denom else 0.0
        candidate = (score, precision, recall, float(cutoff))
        # Prefer higher precision, then higher threshold in a tie.
        if best is None or candidate > best:
            best = candidate
    return {
        "f_beta": float(best[0]),
        "precision": float(best[1]),
        "recall": float(best[2]),
        "cutoff": float(best[3]),
        "beta": float(beta),
    }


def safe_probability_metrics(y_true, probabilities, cutoff):
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    pred = (p >= cutoff).astype(int)
    result = {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()) if len(y) else np.nan,
        "probability_cutoff": float(cutoff),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "average_precision": float(average_precision_score(y, p)) if y.sum() else np.nan,
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "predicted_events": int(pred.sum()),
    }
    result["roc_auc"] = float(roc_auc_score(y, p)) if y.min() != y.max() else np.nan
    return result


# ============================================================
# EXPECTED DART FROM TWO-SIDED REGIME PROBABILITIES
# ============================================================

def combine_regime_probabilities(p_neg, p_pos):
    p_neg = np.clip(np.asarray(p_neg, dtype=float), 0.0, 1.0)
    p_pos = np.clip(np.asarray(p_pos, dtype=float), 0.0, 1.0)
    total = p_neg + p_pos
    scale = np.maximum(total, 1.0)
    p_neg_adj = p_neg / scale
    p_pos_adj = p_pos / scale
    p_normal = np.clip(1.0 - p_neg_adj - p_pos_adj, 0.0, 1.0)
    return p_neg_adj, p_normal, p_pos_adj


def expected_dart(p_neg, p_pos, mean_neg, mean_normal, mean_pos):
    pn, pnorm, pp = combine_regime_probabilities(p_neg, p_pos)
    forecast = pn * mean_neg + pnorm * mean_normal + pp * mean_pos
    return forecast, pn, pnorm, pp


# ============================================================
# METRICS AND OUTPUTS
# ============================================================

def point_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(predicted - actual)),
        "direction_accuracy": float(np.mean(np.sign(actual) == np.sign(predicted))),
    }


def build_prediction_frame(timestamps, hours, actual, p_neg, p_pos,
                           neg_cutoff, pos_cutoff, expected):
    p_neg_adj, p_normal_adj, p_pos_adj = combine_regime_probabilities(p_neg, p_pos)
    out = pd.DataFrame({
        "delivery_timestamp": pd.to_datetime(timestamps).astype(str).to_numpy(),
        "delivery_hour": np.asarray(hours, dtype=int),
        "actual_dart": np.asarray(actual, dtype=float),
        "prob_negative_spike": p_neg_adj,
        "prob_normal_regime": p_normal_adj,
        "prob_positive_spike": p_pos_adj,
        "predicted_negative_spike": (np.asarray(p_neg) >= neg_cutoff).astype(int),
        "predicted_positive_spike": (np.asarray(p_pos) >= pos_cutoff).astype(int),
        "predicted_dart": np.asarray(expected, dtype=float),
    })
    out["error"] = out["predicted_dart"] - out["actual_dart"]
    out["absolute_error"] = out["error"].abs()
    out["model"] = MODEL_NAME
    return out.sort_values("delivery_timestamp", kind="stable").reset_index(drop=True)


def monthly_performance(pred: pd.DataFrame) -> pd.DataFrame:
    frame = pred.copy()
    frame["delivery_timestamp"] = pd.to_datetime(frame["delivery_timestamp"])
    frame["month"] = frame["delivery_timestamp"].dt.to_period("M").astype(str)
    rows = []
    for month, group in frame.groupby("month", sort=True):
        rows.append({
            "month": month,
            **point_metrics(group["actual_dart"], group["predicted_dart"]),
        })
    return pd.DataFrame(rows)


def update_comparison(test_metrics: dict[str, Any]) -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    path = COMPARISON_DIR / "point_forecast_model_comparison.csv"
    row = {
        "model": MODEL_NAME,
        "n": test_metrics["point_forecast"]["n"],
        "mae": test_metrics["point_forecast"]["mae"],
        "rmse": test_metrics["point_forecast"]["rmse"],
        "bias": test_metrics["point_forecast"]["bias"],
        "negative_spike_average_precision": test_metrics["negative_spike"]["average_precision"],
        "positive_spike_average_precision": test_metrics["positive_spike"]["average_precision"],
    }
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not existing.empty:
        existing = existing[existing["model"] != MODEL_NAME]
    comparison = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    if "mae" in comparison:
        comparison = comparison.sort_values("mae", na_position="last")
    comparison.to_csv(path, index=False)


# ============================================================
# MAIN MODEL
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Two-sided DART spike logistic model")
    parser.add_argument("--input", default=None, help="Processed DART feature CSV")
    parser.add_argument("--spike-quantile", type=float, default=SPIKE_QUANTILE,
                        help="Training-tail quantile for positive and negative spike thresholds")
    args = parser.parse_args()

    if not 0.80 <= args.spike_quantile <= 0.995:
        raise ValueError("--spike-quantile must be between 0.80 and 0.995")

    input_path = resolve_input(args.input)
    X, y, timestamps, dates, hours, feature_audit, metadata = load_data(input_path)

    train_dates, validation_dates, test_dates = chronological_dates(dates)
    train_mask = dates.isin(train_dates)
    val_mask = dates.isin(validation_dates)
    test_mask = dates.isin(test_dates)

    # D-2/D-7/D-14 lags must all exist. This removes the initial lag warm-up,
    # but does not shorten the validation or test date windows.
    lag_cols = ["dart_lag_2d_same_hour", "dart_lag_7d_same_hour", "dart_lag_14d_same_hour"]
    lag_ok = X[lag_cols].notna().all(axis=1)
    train_mask &= lag_ok
    val_mask &= lag_ok
    test_mask &= lag_ok

    # Confirm all planned validation/test days remain complete.
    for label, mask, expected_dates in [
        ("validation", val_mask, validation_dates),
        ("test", test_mask, test_dates),
    ]:
        counts = pd.DataFrame({"date": dates.loc[mask], "hour": hours.loc[mask]}).groupby("date")["hour"].nunique()
        if len(counts) != len(expected_dates) or not counts.eq(24).all():
            raise ValueError(f"{label.title()} does not retain 24 observations on every planned market day.")

    # Retain numeric predictors with sufficient training-period support.
    train_missing = X.loc[train_mask].isna().mean()
    feature_cols = [
        c for c in X.columns
        if train_missing[c] <= 0.40 and X.loc[train_mask, c].nunique(dropna=True) > 1
    ]
    if not feature_cols:
        raise ValueError("No eligible predictors remain after training-only filtering.")

    X_model = X[feature_cols].copy()

    # Asymmetric positive and negative thresholds, determined from training only.
    y_train = y.loc[train_mask]
    positive_train = y_train[y_train > 0]
    negative_train = y_train[y_train < 0]
    if len(positive_train) < 20 or len(negative_train) < 20:
        raise ValueError("Insufficient positive or negative training observations to define spike thresholds.")
    gamma_pos = float(positive_train.quantile(args.spike_quantile))
    gamma_neg = float(negative_train.abs().quantile(args.spike_quantile))

    y_pos = (y >= gamma_pos).astype(int)
    y_neg = (y <= -gamma_neg).astype(int)

    if y_pos.loc[train_mask].nunique() < 2 or y_neg.loc[train_mask].nunique() < 2:
        raise ValueError("A spike label contains only one class in training. Lower --spike-quantile.")

    X_train = X_model.loc[train_mask].reset_index(drop=True)
    X_val = X_model.loc[val_mask].reset_index(drop=True)
    X_test = X_model.loc[test_mask].reset_index(drop=True)
    y_pos_train = y_pos.loc[train_mask].reset_index(drop=True)
    y_neg_train = y_neg.loc[train_mask].reset_index(drop=True)
    y_pos_val = y_pos.loc[val_mask].reset_index(drop=True)
    y_neg_val = y_neg.loc[val_mask].reset_index(drop=True)
    y_pos_test = y_pos.loc[test_mask].reset_index(drop=True)
    y_neg_test = y_neg.loc[test_mask].reset_index(drop=True)

    train_date_rows = dates.loc[train_mask].reset_index(drop=True)
    val_date_rows = dates.loc[val_mask].reset_index(drop=True)
    test_date_rows = dates.loc[test_mask].reset_index(drop=True)

    # Inner expanding-window CV chooses regularisation using training period only.
    best_C_neg, cv_neg = tune_logistic_C(X_train, y_neg_train, train_date_rows, feature_cols, "negative")
    best_C_pos, cv_pos = tune_logistic_C(X_train, y_pos_train, train_date_rows, feature_cols, "positive")

    # Validation-only threshold selection.
    negative_validation_model = make_pipeline(best_C_neg)
    positive_validation_model = make_pipeline(best_C_pos)
    convergence_warnings = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        negative_validation_model.fit(X_train, y_neg_train)
        positive_validation_model.fit(X_train, y_pos_train)
    convergence_warnings += sum(issubclass(w.category, ConvergenceWarning) for w in caught)

    p_neg_val = negative_validation_model.predict_proba(X_val)[:, 1]
    p_pos_val = positive_validation_model.predict_proba(X_val)[:, 1]
    neg_cut = choose_probability_cutoff(y_neg_val, p_neg_val, beta=1.0)
    pos_cut = choose_probability_cutoff(y_pos_val, p_pos_val, beta=1.0)

    # Refit classifiers on train + validation, then score untouched test.
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_neg_train_val = pd.concat([y_neg_train, y_neg_val], ignore_index=True)
    y_pos_train_val = pd.concat([y_pos_train, y_pos_val], ignore_index=True)

    negative_model = make_pipeline(best_C_neg)
    positive_model = make_pipeline(best_C_pos)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        negative_model.fit(X_train_val, y_neg_train_val)
        positive_model.fit(X_train_val, y_pos_train_val)
    convergence_warnings += sum(issubclass(w.category, ConvergenceWarning) for w in caught)

    p_neg_test_raw = negative_model.predict_proba(X_test)[:, 1]
    p_pos_test_raw = positive_model.predict_proba(X_test)[:, 1]

    # Regime means are estimated from train + validation? No: to keep the
    # test-stage regime severity estimates fixed before seeing test outcomes,
    # use training-period values only.
    train_y = y.loc[train_mask].to_numpy(dtype=float)
    normal_values = train_y[(train_y > -gamma_neg) & (train_y < gamma_pos)]
    positive_values = train_y[train_y >= gamma_pos]
    negative_values = train_y[train_y <= -gamma_neg]
    mean_normal = float(np.mean(normal_values)) if len(normal_values) else 0.0
    mean_positive = float(np.mean(positive_values))
    mean_negative = float(np.mean(negative_values))

    # Validation expected-DART forecast uses validation classifier fitted on
    # training only; test forecast uses refitted classifiers and training means.
    p_neg_val_raw = negative_validation_model.predict_proba(X_val)[:, 1]
    p_pos_val_raw = positive_validation_model.predict_proba(X_val)[:, 1]
    expected_val, _, _, _ = expected_dart(
        p_neg_val_raw, p_pos_val_raw, mean_negative, mean_normal, mean_positive
    )
    expected_test, _, _, _ = expected_dart(
        p_neg_test_raw, p_pos_test_raw, mean_negative, mean_normal, mean_positive
    )

    t_val = timestamps.loc[val_mask].reset_index(drop=True)
    h_val = hours.loc[val_mask].reset_index(drop=True)
    y_val = y.loc[val_mask].reset_index(drop=True)
    t_test = timestamps.loc[test_mask].reset_index(drop=True)
    h_test = hours.loc[test_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)

    validation_predictions = build_prediction_frame(
        t_val, h_val, y_val, p_neg_val_raw, p_pos_val_raw,
        neg_cut["cutoff"], pos_cut["cutoff"], expected_val
    )
    test_predictions = build_prediction_frame(
        t_test, h_test, y_test, p_neg_test_raw, p_pos_test_raw,
        neg_cut["cutoff"], pos_cut["cutoff"], expected_test
    )

    validation_metrics = {
        "model": MODEL_NAME,
        "point_forecast": point_metrics(y_val, expected_val),
        "negative_spike": safe_probability_metrics(y_neg_val, p_neg_val_raw, neg_cut["cutoff"]),
        "positive_spike": safe_probability_metrics(y_pos_val, p_pos_val_raw, pos_cut["cutoff"]),
        "negative_probability_cutoff": neg_cut,
        "positive_probability_cutoff": pos_cut,
    }
    test_metrics = {
        "model": MODEL_NAME,
        "point_forecast": point_metrics(y_test, expected_test),
        "negative_spike": safe_probability_metrics(y_neg_test, p_neg_test_raw, neg_cut["cutoff"]),
        "positive_spike": safe_probability_metrics(y_pos_test, p_pos_test_raw, pos_cut["cutoff"]),
        "negative_probability_cutoff": neg_cut,
        "positive_probability_cutoff": pos_cut,
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    feature_audit.to_csv(RESULT_DIR / "feature_audit.csv", index=False)
    validation_predictions.to_csv(RESULT_DIR / "validation_predictions.csv", index=False)
    test_predictions.to_csv(RESULT_DIR / "test_predictions.csv", index=False)
    monthly_performance(test_predictions).to_csv(RESULT_DIR / "monthly_test_performance.csv", index=False)
    pd.DataFrame([{
        "model": MODEL_NAME,
        **{f"point_{k}": v for k, v in validation_metrics["point_forecast"].items()},
        **{f"negative_{k}": v for k, v in validation_metrics["negative_spike"].items() if k != "n"},
        **{f"positive_{k}": v for k, v in validation_metrics["positive_spike"].items() if k != "n"},
    }]).to_csv(RESULT_DIR / "validation_metrics.csv", index=False)
    pd.DataFrame([{
        "model": MODEL_NAME,
        **{f"point_{k}": v for k, v in test_metrics["point_forecast"].items()},
        **{f"negative_{k}": v for k, v in test_metrics["negative_spike"].items() if k != "n"},
        **{f"positive_{k}": v for k, v in test_metrics["positive_spike"].items() if k != "n"},
    }]).to_csv(RESULT_DIR / "test_metrics.csv", index=False)

    cv_scores = pd.concat([cv_neg, cv_pos], ignore_index=True)
    cv_scores.to_csv(RESULT_DIR / "regularisation_cv_scores.csv", index=False)

    # Save coefficients as model diagnostics.
    coefficient_rows = []
    for side, fitted in [("negative", negative_model), ("positive", positive_model)]:
        logistic = fitted.named_steps["logistic"]
        for feature, coef in zip(feature_cols, logistic.coef_[0]):
            coefficient_rows.append({
                "side": side,
                "feature": feature,
                "coefficient_standardized": float(coef),
                "absolute_coefficient": float(abs(coef)),
            })
    coef_frame = pd.DataFrame(coefficient_rows)
    coef_frame.to_csv(RESULT_DIR / "feature_importance.csv", index=False)

    joblib.dump({
        "model": MODEL_NAME,
        "negative_model": negative_model,
        "positive_model": positive_model,
        "features": feature_cols,
        "gamma_negative": gamma_neg,
        "gamma_positive": gamma_pos,
        "negative_probability_cutoff": neg_cut["cutoff"],
        "positive_probability_cutoff": pos_cut["cutoff"],
        "training_regime_means": {
            "negative": mean_negative,
            "normal": mean_normal,
            "positive": mean_positive,
        },
        "forecast_origin": "D-2 intended; verify feature vintages in feature_audit.csv",
    }, RESULT_DIR / "model.joblib")

    summary = {
        "model": MODEL_NAME,
        "input_file": str(input_path),
        "target_column": metadata["target_column"],
        "date_column": metadata["date_column"],
        "hour_column": metadata["hour_column"],
        "raw_market_days": int(dates.nunique()),
        "raw_hourly_observations": int(len(y)),
        "training_market_days": int(len(train_dates)),
        "validation_market_days": int(len(validation_dates)),
        "test_market_days": int(len(test_dates)),
        "train_start": str(train_dates.min()),
        "train_end": str(train_dates.max()),
        "validation_start": str(validation_dates.min()),
        "validation_end": str(validation_dates.max()),
        "test_start": str(test_dates.min()),
        "test_end": str(test_dates.max()),
        "n_train_after_lags": int(train_mask.sum()),
        "n_validation_after_lags": int(val_mask.sum()),
        "n_test_after_lags": int(test_mask.sum()),
        "n_features": int(len(feature_cols)),
        "features": feature_cols,
        "spike_quantile": float(args.spike_quantile),
        "gamma_negative": gamma_neg,
        "gamma_positive": gamma_pos,
        "best_C_negative": best_C_neg,
        "best_C_positive": best_C_pos,
        "negative_probability_cutoff": neg_cut,
        "positive_probability_cutoff": pos_cut,
        "convergence_warnings": int(convergence_warnings),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "note": (
            "Separate positive/negative logistic spike classifiers. "
            "C selected using expanding chronological CV on training data; "
            "probability cutoffs selected on validation data only."
        ),
    }
    with open(RESULT_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    update_comparison(test_metrics)

    print("=" * 76)
    print("ONTARIO DART: TWO-SIDED SPIKE LOGISTIC MODEL")
    print("=" * 76)
    print(f"Input: {input_path}")
    print(f"Complete market days: {dates.nunique():,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Train / validation / test: {train_mask.sum():,} / {val_mask.sum():,} / {test_mask.sum():,}")
    print(f"Test dates: {test_dates.min().date()} to {test_dates.max().date()}")
    print(f"Negative spike threshold: {gamma_neg:.4f}")
    print(f"Positive spike threshold: {gamma_pos:.4f}")
    print(f"Selected C (negative / positive): {best_C_neg:.4g} / {best_C_pos:.4g}")
    print("\nTEST POINT-FORECAST METRICS")
    for key, value in test_metrics["point_forecast"].items():
        if key != "n":
            print(f"{key}: {value:.4f}")
    print("\nTEST NEGATIVE-SPIKE METRICS")
    for key, value in test_metrics["negative_spike"].items():
        print(f"{key}: {value}")
    print("\nTEST POSITIVE-SPIKE METRICS")
    for key, value in test_metrics["positive_spike"].items():
        print(f"{key}: {value}")
    print(f"Results: {RESULT_DIR}")
    print("Review feature_audit.csv for actual D-2 information availability.")
    print("=" * 76)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
