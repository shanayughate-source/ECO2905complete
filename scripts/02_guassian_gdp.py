#!/usr/bin/env python3
"""Standalone Ontario DART covariate-dependent Gaussian-GPD model.

This is one of the ONLY TWO Python scripts needed in scripts/.
It builds the ONZP DART target and leakage-controlled historical features
from raw IESO CSV files, then trains and backtests the three-regime model:
negative GPD tail, truncated-Gaussian centre, and positive GPD tail.

Inputs in datasets/raw/: da_ontario_zonal.csv and dart_ontario_zonal.csv
(alternative filenames are detected automatically).
Run: .venv/bin/python scripts/02_gaussian_gpd.py
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import timedelta, timezone
from pathlib import Path
from typing import Dict, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import genpareto, norm, truncnorm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
RAW = DATASETS / "raw"
PROCESSED = DATASETS / "processed"
RESULTS = ROOT / "results"
TARGET_COLUMN = "target_dart"
SPIKE_THRESHOLD = 100.0
MARKET_TZ = timezone(timedelta(hours=-5), name="EST")
TORONTO_TZ = "America/Toronto"

FEATURES = [
    "hour", "day_of_week", "is_weekend", "month",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "year_sin", "year_cos",
    "dart_same_hour_d2", "last_observed_same_hour_dart",
    "last_observed_same_hour_da", "last_observed_same_hour_rt",
    "dart_mean_7d", "dart_median_7d", "dart_std_7d",
    "dart_abs_mean_7d", "dart_spike_rate_7d",
    "dart_positive_rate_7d", "dart_negative_rate_7d",
    "da_mean_7d", "rt_mean_7d", "dart_observations_7d",
    "dart_mean_28d", "dart_median_28d", "dart_std_28d",
    "dart_abs_mean_28d", "dart_spike_rate_28d",
    "da_mean_28d", "rt_mean_28d", "dart_observations_28d",
    "daily_mean_dart_7d", "daily_abs_mean_dart_7d",
    "daily_max_abs_dart_7d", "daily_spike_hours_7d",
    "daily_mean_dart_28d", "daily_abs_mean_dart_28d",
    "daily_max_abs_dart_28d",
]


def find_raw_file(candidates, label):
    for path in candidates:
        if path.exists():
            return path
    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find the {label} CSV. Checked:\n{checked}\n"
        "Place the original IESO CSV files in datasets/raw/."
    )


def read_price_file(path: Path, label: str) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    required = {"location", "lmp"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")

    if "interval_start_utc" in data.columns:
        ts = pd.to_datetime(data["interval_start_utc"], utc=True, errors="coerce")
    elif "interval_start_local" in data.columns:
        local = pd.to_datetime(data["interval_start_local"], errors="coerce")
        if local.dt.tz is None:
            ts = local.dt.tz_localize(MARKET_TZ).dt.tz_convert("UTC")
        else:
            ts = local.dt.tz_convert("UTC")
    else:
        raise ValueError(
            f"{label} needs interval_start_utc or interval_start_local. "
            f"Available columns: {list(data.columns)}"
        )

    data = data.copy()
    data["interval_start_utc"] = ts
    data["location"] = data["location"].astype(str).str.upper().str.strip()
    data["lmp"] = pd.to_numeric(data["lmp"], errors="coerce")
    data = data.dropna(subset=["interval_start_utc", "lmp"])
    data = data[data["location"] == "ONZP"].copy()
    if data.empty:
        raise ValueError(f"No ONZP rows were found in {label}.")
    return data


def build_target_and_features() -> pd.DataFrame:
    """Create DART and its 39 leakage-controlled historical predictors."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    da_path = find_raw_file([
        RAW / "da_ontario_zonal.csv",
        RAW / "da_ontario_zonal_hourly.csv",
        DATASETS / "da_ontario_zonal.csv",
    ], "day-ahead Ontario zonal price")
    rt_path = find_raw_file([
        RAW / "dart_ontario_zonal.csv",
        RAW / "rt_ontario_zonal_5min.csv",
        DATASETS / "dart_ontario_zonal.csv",
        DATASETS / "rt_ontario_zonal_5min.csv",
    ], "real-time Ontario zonal price")

    print("DA input:", da_path)
    print("RT input:", rt_path)
    da = read_price_file(da_path, "day-ahead price file")
    rt = read_price_file(rt_path, "real-time price file")

    da["hour_start_utc"] = da["interval_start_utc"].dt.floor("h")
    if da.duplicated("hour_start_utc").any():
        raise ValueError("Duplicate day-ahead ONZP hourly observations were found.")
    da_hourly = da[["hour_start_utc", "lmp"]].rename(columns={"lmp": "da_lmp"})

    rt["hour_start_utc"] = rt["interval_start_utc"].dt.floor("h")
    if rt.duplicated("interval_start_utc").any():
        raise ValueError("Duplicate real-time five-minute timestamps were found.")
    rt_hourly = rt.groupby("hour_start_utc").agg(
        rt_lmp=("lmp", "mean"),
        rt_count=("lmp", "count"),
    ).reset_index()

    hourly = da_hourly.merge(rt_hourly, on="hour_start_utc", how="inner", validate="one_to_one")
    hourly["rt_count"] = hourly["rt_count"].fillna(0).astype(int)
    local = hourly["hour_start_utc"].dt.tz_convert(MARKET_TZ)
    hourly["market_date"] = pd.to_datetime(local.dt.strftime("%Y-%m-%d"))
    hourly["market_hour"] = local.dt.hour.astype(int)
    hourly[TARGET_COLUMN] = hourly["da_lmp"] - hourly["rt_lmp"]
    hourly["hour_complete"] = (hourly["rt_count"] == 12) & hourly[TARGET_COLUMN].notna()

    all_dates = pd.date_range(hourly["market_date"].min(), hourly["market_date"].max(), freq="D")
    complete_dates = []
    for market_date in all_dates:
        one_day = hourly[hourly["market_date"] == market_date]
        hours = sorted(one_day["market_hour"].unique().tolist())
        complete = (
            len(one_day) == 24
            and hours == list(range(24))
            and bool(one_day["hour_complete"].all())
        )
        if complete:
            complete_dates.append(market_date)

    target = hourly[
        hourly["market_date"].isin(complete_dates) & hourly["hour_complete"]
    ][["market_date", "market_hour", "da_lmp", "rt_lmp", TARGET_COLUMN]].copy()
    target = target.sort_values(["market_date", "market_hour"]).reset_index(drop=True)
    if target.empty:
        raise ValueError("No complete 24-hour market days were found.")
    if not (target.groupby("market_date").size() == 24).all():
        raise ValueError("The complete-day filter did not produce 24 rows per market date.")

    # Daily features use only information available by D-2 at the D-1 09:00 origin.
    start_date = target["market_date"].min()
    end_date = target["market_date"].max()
    calendar = pd.date_range(start_date, end_date, freq="D")
    daily = target.groupby("market_date").agg(
        daily_mean_dart=(TARGET_COLUMN, "mean"),
        daily_abs_mean_dart=(TARGET_COLUMN, lambda values: values.abs().mean()),
        daily_max_abs_dart=(TARGET_COLUMN, lambda values: values.abs().max()),
        daily_spike_hours=(TARGET_COLUMN, lambda values: int((values.abs() >= SPIKE_THRESHOLD).sum())),
    ).reindex(calendar)
    daily_lagged = daily.shift(2)
    daily_7 = daily_lagged.rolling(7, min_periods=1)
    daily_28 = daily_lagged.rolling(28, min_periods=1)
    daily_features = pd.DataFrame({
        "market_date": calendar,
        "daily_mean_dart_7d": daily_7["daily_mean_dart"].mean().to_numpy(),
        "daily_abs_mean_dart_7d": daily_7["daily_abs_mean_dart"].mean().to_numpy(),
        "daily_max_abs_dart_7d": daily_7["daily_max_abs_dart"].max().to_numpy(),
        "daily_spike_hours_7d": daily_7["daily_spike_hours"].mean().to_numpy(),
        "daily_mean_dart_28d": daily_28["daily_mean_dart"].mean().to_numpy(),
        "daily_abs_mean_dart_28d": daily_28["daily_abs_mean_dart"].mean().to_numpy(),
        "daily_max_abs_dart_28d": daily_28["daily_max_abs_dart"].max().to_numpy(),
    })

    hour_frames = []
    for hour in range(24):
        one_hour = target[target["market_hour"] == hour].set_index("market_date")
        dart = one_hour[TARGET_COLUMN].reindex(calendar)
        da_series = one_hour["da_lmp"].reindex(calendar)
        rt_series = one_hour["rt_lmp"].reindex(calendar)
        dart_lag = dart.shift(2)
        da_lag = da_series.shift(2)
        rt_lag = rt_series.shift(2)
        dart_latest = dart_lag.ffill()
        da_latest = da_lag.ffill()
        rt_latest = rt_lag.ffill()

        dart_7 = dart_lag.rolling(7, min_periods=1)
        dart_28 = dart_lag.rolling(28, min_periods=1)
        da_7 = da_lag.rolling(7, min_periods=1)
        rt_7 = rt_lag.rolling(7, min_periods=1)
        da_28 = da_lag.rolling(28, min_periods=1)
        rt_28 = rt_lag.rolling(28, min_periods=1)
        spike_lag = (dart_lag.abs() >= SPIKE_THRESHOLD).where(dart_lag.notna())
        positive_lag = (dart_lag > 0).where(dart_lag.notna())
        negative_lag = (dart_lag < 0).where(dart_lag.notna())

        one_frame = pd.DataFrame({
            "market_date": calendar,
            "market_hour": hour,
            "dart_same_hour_d2": dart_lag.to_numpy(),
            "last_observed_same_hour_dart": dart_latest.to_numpy(),
            "last_observed_same_hour_da": da_latest.to_numpy(),
            "last_observed_same_hour_rt": rt_latest.to_numpy(),
            "dart_mean_7d": dart_7.mean().to_numpy(),
            "dart_median_7d": dart_7.median().to_numpy(),
            "dart_std_7d": dart_7.std().to_numpy(),
            "dart_abs_mean_7d": dart_lag.abs().rolling(7, min_periods=1).mean().to_numpy(),
            "dart_spike_rate_7d": spike_lag.rolling(7, min_periods=1).mean().to_numpy(),
            "dart_positive_rate_7d": positive_lag.rolling(7, min_periods=1).mean().to_numpy(),
            "dart_negative_rate_7d": negative_lag.rolling(7, min_periods=1).mean().to_numpy(),
            "da_mean_7d": da_7.mean().to_numpy(),
            "rt_mean_7d": rt_7.mean().to_numpy(),
            "dart_observations_7d": dart_7.count().to_numpy(),
            "dart_mean_28d": dart_28.mean().to_numpy(),
            "dart_median_28d": dart_28.median().to_numpy(),
            "dart_std_28d": dart_28.std().to_numpy(),
            "dart_abs_mean_28d": dart_lag.abs().rolling(28, min_periods=1).mean().to_numpy(),
            "dart_spike_rate_28d": spike_lag.rolling(28, min_periods=1).mean().to_numpy(),
            "da_mean_28d": da_28.mean().to_numpy(),
            "rt_mean_28d": rt_28.mean().to_numpy(),
            "dart_observations_28d": dart_28.count().to_numpy(),
        })
        hour_frames.append(one_frame)

    features = pd.concat(hour_frames, ignore_index=True)
    features = features.merge(daily_features, on="market_date", how="left", validate="many_to_one")
    features["hour"] = features["market_hour"]
    features["day_of_week"] = features["market_date"].dt.dayofweek
    features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)
    features["month"] = features["market_date"].dt.month
    features["hour_sin"] = np.sin(2.0 * np.pi * features["hour"] / 24.0)
    features["hour_cos"] = np.cos(2.0 * np.pi * features["hour"] / 24.0)
    features["weekday_sin"] = np.sin(2.0 * np.pi * features["day_of_week"] / 7.0)
    features["weekday_cos"] = np.cos(2.0 * np.pi * features["day_of_week"] / 7.0)
    day_of_year = features["market_date"].dt.dayofyear
    days_in_year = np.where(features["market_date"].dt.is_leap_year, 366, 365)
    features["year_sin"] = np.sin(2.0 * np.pi * (day_of_year - 1) / days_in_year)
    features["year_cos"] = np.cos(2.0 * np.pi * (day_of_year - 1) / days_in_year)

    output_columns = ["market_date", "market_hour", TARGET_COLUMN, "da_lmp", "rt_lmp"] + FEATURES
    result = target.merge(
        features[["market_date", "market_hour"] + FEATURES],
        on=["market_date", "market_hour"],
        how="left",
        validate="one_to_one",
    )
    result = result[output_columns].sort_values(["market_date", "market_hour"]).reset_index(drop=True)
    target.to_csv(PROCESSED / "dart_target_complete_market_days.csv", index=False)
    result.to_csv(PROCESSED / "dart_historical_features.csv", index=False)

    print("Complete market days:", result["market_date"].nunique())
    print("Hourly observations:", len(result))
    print("Feature count:", len(FEATURES))
    print("Date range:", result["market_date"].min().date(), "to", result["market_date"].max().date())
    print("Saved processed target:", PROCESSED / "dart_target_complete_market_days.csv")
    print("Saved processed features:", PROCESSED / "dart_historical_features.csv")
    return result


def chronological_split(frame: pd.DataFrame, validation_fraction=0.15, test_fraction=0.15):
    dates = np.array(sorted(frame["market_date"].dropna().unique()))
    if len(dates) < 90:
        warnings.warn(f"Only {len(dates)} complete market days are available; estimates may be unstable.")
    n_test = max(1, int(round(len(dates) * test_fraction)))
    n_validation = max(1, int(round(len(dates) * validation_fraction)))
    n_train = len(dates) - n_validation - n_test
    if n_train < 30:
        raise ValueError("Not enough market days for the chronological train/validation/test split.")
    validation_dates = dates[n_train:n_train + n_validation]
    test_dates = dates[n_train + n_validation:]
    validation_start = pd.Timestamp(validation_dates[0])
    test_start = pd.Timestamp(test_dates[0])
    validation = frame[frame["market_date"].isin(validation_dates)].copy()
    test = frame[frame["market_date"].isin(test_dates)].copy()
    validation_training = frame[frame["market_date"] <= validation_start - pd.Timedelta(days=2)].copy()
    test_training = frame[frame["market_date"] <= test_start - pd.Timedelta(days=2)].copy()
    if validation_training.empty or test_training.empty:
        raise ValueError("Not enough prior observations after applying the D-2 forecast information cutoff.")
    return {
        "validation": validation,
        "test": test,
        "validation_training": validation_training,
        "test_training": test_training,
        "validation_cutoff": validation_start - pd.Timedelta(days=2),
        "test_cutoff": test_start - pd.Timedelta(days=2),
        "n_dates": len(dates),
    }


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    actual_spike = np.abs(actual) >= SPIKE_THRESHOLD
    predicted_spike = np.abs(predicted) >= SPIKE_THRESHOLD
    true_positive = int(np.sum(actual_spike & predicted_spike))
    false_positive = int(np.sum(~actual_spike & predicted_spike))
    false_negative = int(np.sum(actual_spike & ~predicted_spike))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "Bias": float(np.mean(error)),
        "Spike_precision": float(true_positive / (true_positive + false_positive)) if true_positive + false_positive else 0.0,
        "Spike_recall": float(true_positive / (true_positive + false_negative)) if true_positive + false_negative else 0.0,
        "Spike_MAE": float(np.mean(np.abs(error[actual_spike]))) if actual_spike.any() else float("nan"),
        "Observations": int(len(actual)),
    }


def write_model_outputs(model_name, result_dir, validation_frame, validation_prediction,
                        validation_metrics, test_frame, test_prediction, test_metrics,
                        validation_extra=None, test_extra=None, model_artifact=None):
    result_dir.mkdir(parents=True, exist_ok=True)
    for label, actual_frame, prediction, extra in [
        ("validation", validation_frame, validation_prediction, validation_extra),
        ("test", test_frame, test_prediction, test_extra),
    ]:
        output = actual_frame[["market_date", "market_hour", TARGET_COLUMN]].copy()
        output = output.rename(columns={TARGET_COLUMN: "actual_dart"})
        output["predicted_dart"] = np.asarray(prediction, dtype=float)
        output["error"] = output["predicted_dart"] - output["actual_dart"]
        output["absolute_error"] = output["error"].abs()
        if extra is not None:
            extra = extra.reset_index(drop=True)
            for column in extra.columns:
                if column != "predicted_dart":
                    output[column] = extra[column].to_numpy()
        output.to_csv(result_dir / f"{label}_predictions.csv", index=False)

    pd.DataFrame([validation_metrics]).to_csv(result_dir / "validation_metrics.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(result_dir / "test_metrics.csv", index=False)

    monthly = pd.DataFrame({
        "market_date": test_frame["market_date"].to_numpy(),
        "actual_dart": test_frame[TARGET_COLUMN].to_numpy(dtype=float),
        "predicted_dart": np.asarray(test_prediction, dtype=float),
    })
    monthly["month"] = pd.to_datetime(monthly["market_date"]).dt.to_period("M").astype(str)
    rows = []
    for month, group in monthly.groupby("month"):
        error = group["predicted_dart"].to_numpy() - group["actual_dart"].to_numpy()
        rows.append({
            "month": month,
            "observations": len(group),
            "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "Bias": float(np.mean(error)),
            "actual_spike_hours": int((np.abs(group["actual_dart"]) >= SPIKE_THRESHOLD).sum()),
        })
    pd.DataFrame(rows).to_csv(result_dir / "monthly_test_performance.csv", index=False)

    summary = {
        "model": model_name,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    with open(result_dir / "model_summary.json", "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, allow_nan=True)
    if model_artifact is not None:
        import joblib
        joblib.dump(model_artifact, result_dir / "model.joblib")


def update_comparison_table():
    comparison_dir = RESULTS / "model_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    model_folders = [
        ("Hour_of_week_median", RESULTS / "hour_of_week_median"),
        ("Gaussian_GPD", RESULTS / "gaussian_gpd"),
    ]
    for split_name in ["validation", "test"]:
        rows = []
        for model_name, model_dir in model_folders:
            metrics_path = model_dir / f"{split_name}_metrics.csv"
            if metrics_path.exists():
                metrics = pd.read_csv(metrics_path)
                if len(metrics):
                    row = metrics.iloc[0].to_dict()
                    row["Model"] = model_name
                    rows.append(row)
        if rows:
            comparison = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
            comparison.to_csv(comparison_dir / f"{split_name}_comparison.csv", index=False)


def fit_gpd_shape(excess: np.ndarray) -> tuple[float, float]:
    excess = np.asarray(excess, dtype=float)
    excess = excess[np.isfinite(excess) & (excess > 0)]
    if len(excess) < 40:
        raise ValueError(f"Only {len(excess)} observations in a GPD tail; at least 40 are required.")
    try:
        initial_shape, _, _ = genpareto.fit(excess, floc=0.0)
        shape = float(np.clip(initial_shape, -0.25, 0.70))
        shape, _, scale = genpareto.fit(excess, f0=shape, floc=0.0)
        return float(shape), float(scale)
    except Exception as exc:
        warnings.warn(f"GPD maximum-likelihood fit was unstable ({exc}); using a moment fallback.")
        mean_excess = float(np.mean(excess))
        variance = float(np.var(excess, ddof=1))
        shape = 0.0 if variance <= 0 else 0.5 * (1.0 - mean_excess ** 2 / variance)
        shape = float(np.clip(shape, -0.25, 0.70))
        scale = max(1e-6, mean_excess * (1.0 - shape))
        return shape, scale


class GaussianGPDThreeRegime:
    """Truncated Gaussian centre with separate covariate-scaled GPD tails."""

    def __init__(self, lower_quantile=0.05, upper_quantile=0.95, random_state=42):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Sequence[float]):
        y = np.asarray(y, dtype=float)
        if not 0 < self.lower_quantile < self.upper_quantile < 1:
            raise ValueError("Require 0 < lower_quantile < upper_quantile < 1.")
        if len(y) != len(X) or not np.isfinite(y).all():
            raise ValueError("X and y must have equal length and finite targets.")

        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        Xi = self.imputer_.fit_transform(X)
        self.lower_ = float(np.quantile(y, self.lower_quantile))
        self.upper_ = float(np.quantile(y, self.upper_quantile))
        labels = np.where(y < self.lower_, -1, np.where(y > self.upper_, 1, 0))
        counts = {int(label): int(np.sum(labels == label)) for label in [-1, 0, 1]}
        if min(counts.values()) < 40:
            raise ValueError(
                f"Too few observations in one regime: {counts}. "
                "Use more history or widen tails, for example --lower-quantile 0.10 --upper-quantile 0.90."
            )
        self.training_regime_counts_ = counts

        self.gate_ = Pipeline([
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(
                C=0.20, max_iter=5000, solver="lbfgs", random_state=self.random_state
            )),
        ])
        self.gate_.fit(Xi, labels)
        self.gate_classes_ = self.gate_.named_steps["logit"].classes_

        normal_mask = labels == 0
        self.normal_mean_ = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.04, max_iter=160,
            max_leaf_nodes=7, min_samples_leaf=35, l2_regularization=4.0,
            random_state=self.random_state,
        )
        self.normal_mean_.fit(Xi[normal_mask], y[normal_mask])
        normal_residual = y[normal_mask] - self.normal_mean_.predict(Xi[normal_mask])
        median_residual = float(np.median(normal_residual))
        robust_sigma = 1.4826 * float(np.median(np.abs(normal_residual - median_residual)))
        if not np.isfinite(robust_sigma) or robust_sigma <= 1e-6:
            robust_sigma = float(np.std(normal_residual, ddof=1))
        self.normal_sigma_ = max(robust_sigma, 1e-3)

        sigma_target = np.maximum(np.abs(normal_residual) * np.sqrt(np.pi / 2.0), 1e-3)
        self.normal_sigma_model_ = HistGradientBoostingRegressor(
            loss="poisson", learning_rate=0.04, max_iter=100,
            max_leaf_nodes=5, min_samples_leaf=40, l2_regularization=6.0,
            random_state=self.random_state,
        )
        self.normal_sigma_model_.fit(Xi[normal_mask], sigma_target)

        negative_mask = labels == -1
        negative_excess = self.lower_ - y[negative_mask]
        self.negative_shape_, self.negative_scale_global_ = fit_gpd_shape(negative_excess)
        self.negative_severity_ = HistGradientBoostingRegressor(
            loss="poisson", learning_rate=0.04, max_iter=120,
            max_leaf_nodes=7, min_samples_leaf=25, l2_regularization=4.0,
            random_state=self.random_state,
        )
        self.negative_severity_.fit(Xi[negative_mask], negative_excess)

        positive_mask = labels == 1
        positive_excess = y[positive_mask] - self.upper_
        self.positive_shape_, self.positive_scale_global_ = fit_gpd_shape(positive_excess)
        self.positive_severity_ = HistGradientBoostingRegressor(
            loss="poisson", learning_rate=0.04, max_iter=120,
            max_leaf_nodes=7, min_samples_leaf=25, l2_regularization=4.0,
            random_state=self.random_state,
        )
        self.positive_severity_.fit(Xi[positive_mask], positive_excess)
        return self

    def _components(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        Xi = self.imputer_.transform(X)
        raw_probabilities = self.gate_.predict_proba(Xi)
        probabilities = np.zeros((len(Xi), 3), dtype=float)
        regime_column = {-1: 0, 0: 1, 1: 2}
        for j, label in enumerate(self.gate_classes_):
            probabilities[:, regime_column[int(label)]] = raw_probabilities[:, j]
        probabilities = np.clip(probabilities, 1e-12, 1.0)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

        mu = self.normal_mean_.predict(Xi)
        sigma_surface = np.maximum(self.normal_sigma_model_.predict(Xi), 1e-3)
        sigma = np.clip(
            0.5 * self.normal_sigma_ + 0.5 * sigma_surface,
            0.5 * self.normal_sigma_, 2.0 * self.normal_sigma_,
        )
        negative_mean = np.maximum(self.negative_severity_.predict(Xi), 1e-6)
        positive_mean = np.maximum(self.positive_severity_.predict(Xi), 1e-6)
        beta_negative = np.maximum(negative_mean * (1.0 - self.negative_shape_), 1e-6)
        beta_positive = np.maximum(positive_mean * (1.0 - self.positive_shape_), 1e-6)
        return {
            "p_negative": probabilities[:, 0],
            "p_normal": probabilities[:, 1],
            "p_positive": probabilities[:, 2],
            "mu": mu,
            "sigma": sigma,
            "beta_negative": beta_negative,
            "beta_positive": beta_positive,
        }

    @staticmethod
    def _truncated_normal_mean(mu, sigma, lower, upper):
        a = (lower - mu) / sigma
        b = (upper - mu) / sigma
        mass = np.maximum(norm.cdf(b) - norm.cdf(a), 1e-12)
        return mu + sigma * (norm.pdf(a) - norm.pdf(b)) / mass

    def _mixture_quantile(self, components, quantile):
        p_negative = components["p_negative"]
        p_normal = components["p_normal"]
        p_positive = components["p_positive"]
        output = np.empty(len(p_negative), dtype=float)
        negative_rows = quantile < p_negative
        normal_rows = (quantile >= p_negative) & (quantile <= p_negative + p_normal)
        positive_rows = ~(negative_rows | normal_rows)

        if negative_rows.any():
            conditional_q = np.clip(quantile / p_negative[negative_rows], 1e-10, 1 - 1e-10)
            excess = genpareto.isf(
                conditional_q, self.negative_shape_, loc=0,
                scale=components["beta_negative"][negative_rows],
            )
            output[negative_rows] = self.lower_ - excess
        if normal_rows.any():
            conditional_q = np.clip(
                (quantile - p_negative[normal_rows]) / p_normal[normal_rows], 1e-9, 1 - 1e-9
            )
            a = (self.lower_ - components["mu"][normal_rows]) / components["sigma"][normal_rows]
            b = (self.upper_ - components["mu"][normal_rows]) / components["sigma"][normal_rows]
            output[normal_rows] = truncnorm.ppf(
                conditional_q, a, b,
                loc=components["mu"][normal_rows], scale=components["sigma"][normal_rows],
            )
        if positive_rows.any():
            conditional_q = np.clip(
                (quantile - p_negative[positive_rows] - p_normal[positive_rows])
                / p_positive[positive_rows], 1e-10, 1 - 1e-10,
            )
            excess = genpareto.ppf(
                conditional_q, self.positive_shape_, loc=0,
                scale=components["beta_positive"][positive_rows],
            )
            output[positive_rows] = self.upper_ + excess
        return output

    def predict_distribution(self, X: pd.DataFrame) -> pd.DataFrame:
        components = self._components(X)
        mean_negative = self.lower_ - components["beta_negative"] / (1.0 - self.negative_shape_)
        mean_normal = self._truncated_normal_mean(
            components["mu"], components["sigma"], self.lower_, self.upper_
        )
        mean_positive = self.upper_ + components["beta_positive"] / (1.0 - self.positive_shape_)
        expected = (
            components["p_negative"] * mean_negative
            + components["p_normal"] * mean_normal
            + components["p_positive"] * mean_positive
        )
        return pd.DataFrame({
            "predicted_dart": expected,
            "predicted_dart_p10": self._mixture_quantile(components, 0.10),
            "predicted_dart_p50": self._mixture_quantile(components, 0.50),
            "predicted_dart_p90": self._mixture_quantile(components, 0.90),
            "prob_negative_spike": components["p_negative"],
            "prob_normal_regime": components["p_normal"],
            "prob_positive_spike": components["p_positive"],
            "prob_any_spike": components["p_negative"] + components["p_positive"],
        }, index=X.index)

    def negative_log_likelihood(self, X: pd.DataFrame, y: Sequence[float]) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        components = self._components(X)
        log_density = np.empty(len(y), dtype=float)
        negative = y < self.lower_
        normal = (y >= self.lower_) & (y <= self.upper_)
        positive = y > self.upper_
        if negative.any():
            excess = self.lower_ - y[negative]
            density = genpareto.logpdf(
                excess, self.negative_shape_, loc=0,
                scale=components["beta_negative"][negative],
            )
            log_density[negative] = np.log(components["p_negative"][negative]) + density
        if normal.any():
            a = (self.lower_ - components["mu"][normal]) / components["sigma"][normal]
            b = (self.upper_ - components["mu"][normal]) / components["sigma"][normal]
            mass = np.maximum(norm.cdf(b) - norm.cdf(a), 1e-12)
            density = norm.logpdf(
                y[normal], loc=components["mu"][normal], scale=components["sigma"][normal]
            ) - np.log(mass)
            log_density[normal] = np.log(components["p_normal"][normal]) + density
        if positive.any():
            excess = y[positive] - self.upper_
            density = genpareto.logpdf(
                excess, self.positive_shape_, loc=0,
                scale=components["beta_positive"][positive],
            )
            log_density[positive] = np.log(components["p_positive"][positive]) + density
        return -np.maximum(log_density, -1e6)


def main():
    parser = argparse.ArgumentParser(description="Train/backtest the Ontario DART Gaussian-GPD model.")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--lower-quantile", type=float, default=0.05)
    parser.add_argument("--upper-quantile", type=float, default=0.95)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 0.4 or not 0 < args.test_fraction < 0.4:
        raise ValueError("Validation and test fractions must be between 0 and 0.4.")
    if not 0 < args.lower_quantile < args.upper_quantile < 1:
        raise ValueError("Require 0 < lower-quantile < upper-quantile < 1.")

    frame = build_target_and_features()
    split = chronological_split(frame, args.validation_fraction, args.test_fraction)
    validation = split["validation"]
    test = split["test"]
    validation_training = split["validation_training"]
    test_training = split["test_training"]

    X_validation = validation[FEATURES].replace([np.inf, -np.inf], np.nan)
    X_test = test[FEATURES].replace([np.inf, -np.inf], np.nan)
    y_validation = validation[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=float)

    validation_model = GaussianGPDThreeRegime(
        args.lower_quantile, args.upper_quantile, args.random_state
    )
    validation_model.fit(validation_training[FEATURES], validation_training[TARGET_COLUMN].to_numpy(dtype=float))
    validation_distribution = validation_model.predict_distribution(X_validation)
    validation_prediction = validation_distribution["predicted_dart"].to_numpy(dtype=float)
    validation_metrics = calculate_metrics(y_validation, validation_prediction)
    validation_metrics["negative_log_likelihood"] = float(
        np.mean(validation_model.negative_log_likelihood(X_validation, y_validation))
    )
    validation_metrics["p10_p90_coverage"] = float(np.mean(
        (y_validation >= validation_distribution["predicted_dart_p10"].to_numpy())
        & (y_validation <= validation_distribution["predicted_dart_p90"].to_numpy())
    ))
    validation_metrics["p10_p90_mean_width"] = float(np.mean(
        validation_distribution["predicted_dart_p90"].to_numpy()
        - validation_distribution["predicted_dart_p10"].to_numpy()
    ))

    test_model = GaussianGPDThreeRegime(
        args.lower_quantile, args.upper_quantile, args.random_state
    )
    test_model.fit(test_training[FEATURES], test_training[TARGET_COLUMN].to_numpy(dtype=float))
    test_distribution = test_model.predict_distribution(X_test)
    test_prediction = test_distribution["predicted_dart"].to_numpy(dtype=float)
    test_metrics = calculate_metrics(y_test, test_prediction)
    test_metrics["negative_log_likelihood"] = float(
        np.mean(test_model.negative_log_likelihood(X_test, y_test))
    )
    test_metrics["p10_p90_coverage"] = float(np.mean(
        (y_test >= test_distribution["predicted_dart_p10"].to_numpy())
        & (y_test <= test_distribution["predicted_dart_p90"].to_numpy())
    ))
    test_metrics["p10_p90_mean_width"] = float(np.mean(
        test_distribution["predicted_dart_p90"].to_numpy()
        - test_distribution["predicted_dart_p10"].to_numpy()
    ))

    final_model = GaussianGPDThreeRegime(
        args.lower_quantile, args.upper_quantile, args.random_state
    )
    final_model.fit(frame[FEATURES], frame[TARGET_COLUMN].to_numpy(dtype=float))
    artifact = {
        "model_name": "Covariate-dependent three-regime Gaussian-GPD",
        "model": final_model,
        "features": FEATURES,
        "target": TARGET_COLUMN,
        "lower_quantile": args.lower_quantile,
        "upper_quantile": args.upper_quantile,
        "forecast_information_cutoff": "previous market day 09:00 Toronto; outcomes through D-2 only",
    }

    result_dir = RESULTS / "gaussian_gpd"
    write_model_outputs(
        "Gaussian-GPD three-regime", result_dir,
        validation, validation_prediction, validation_metrics,
        test, test_prediction, test_metrics,
        validation_extra=validation_distribution,
        test_extra=test_distribution,
        model_artifact=artifact,
    )

    summary_path = result_dir / "model_summary.json"
    with open(summary_path, "r", encoding="utf-8") as file:
        summary = json.load(file)
    summary["validation_training_cutoff"] = str(split["validation_cutoff"].date())
    summary["test_training_cutoff"] = str(split["test_cutoff"].date())
    summary["test_period_start"] = str(test["market_date"].min().date())
    summary["test_period_end"] = str(test["market_date"].max().date())
    summary["training_regime_counts_test_fit"] = test_model.training_regime_counts_
    summary["negative_gpd_shape_test_fit"] = test_model.negative_shape_
    summary["positive_gpd_shape_test_fit"] = test_model.positive_shape_
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=True)

    update_comparison_table()
    print("\n" + "=" * 76)
    print("ONTARIO DART: GAUSSIAN-GPD THREE-REGIME MODEL")
    print("=" * 76)
    print("Market days:", split["n_dates"])
    print("Feature count:", len(FEATURES))
    print("Validation metrics:", validation_metrics)
    print("Test metrics:", test_metrics)
    print("\nSaved all Gaussian-GPD outputs to:", result_dir)
    print("The other script, 01_hour_of_week_median.py, writes to results/hour_of_week_median/.")


if __name__ == "__main__":
    main()
