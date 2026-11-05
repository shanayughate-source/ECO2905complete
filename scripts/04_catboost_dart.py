
#!/usr/bin/env python3
"""CatBoost model for Ontario DART forecasting."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "catboost"
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
    "hour_ending", "hourending", "he", "hour", "hour_of_day"
]

LEAKAGE_TOKENS = [
    "real_time_price", "realtime_price", "rt_price", "rt_lmp", "rt_ozp",
    "real_time_ozp", "actual_real_time", "actual_rt", "realized_rt",
    "actual_dart", "realized_dart", "dart_target", "target_dart",
]

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
RANDOM_SEED = 42


def find_column(columns, aliases):

    lookup = {
        str(column).strip().lower(): str(column)
        for column in columns
    }

    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]

    return None


def resolve_input(input_arg):

    if input_arg:

        path = Path(input_arg).expanduser()

        if not path.is_absolute():
            path = ROOT / path

        if not path.exists():
            raise FileNotFoundError(
                f"Input feature CSV does not exist: {path}"
            )

        return path

    for path in DEFAULT_INPUTS:

        if path.exists():
            return path

    checked = "\n".join(
        f"  - {path}" for path in DEFAULT_INPUTS
    )

    raise FileNotFoundError(
        "Could not find a processed DART feature CSV. Checked:\n"
        f"{checked}\nPass the file location with --input."
    )


def load_data(path):

    raw = pd.read_csv(path, low_memory=False)

    if raw.empty:
        raise ValueError(f"Input CSV is empty: {path}")

    target_col = find_column(
        list(raw.columns), TARGET_ALIASES
    )

    date_col = find_column(
        list(raw.columns), DATE_ALIASES
    )

    hour_col = find_column(
        list(raw.columns), HOUR_ALIASES
    )

    if target_col is None:
        raise ValueError(
            f"No DART target found. Columns: {list(raw.columns)}"
        )

    if date_col is None:
        raise ValueError(
            f"No delivery-date column found. Columns: {list(raw.columns)}"
        )

    timestamps = pd.to_datetime(
        raw[date_col],
        errors="coerce"
    )

    # Convert delivery date and hour ending into hourly timestamps.

    if hour_col is not None and timestamps.notna().any():

        hours = pd.to_numeric(
            raw[hour_col],
            errors="coerce"
        )

        if (
            timestamps.dt.hour.fillna(0).eq(0).all()
            and hours.between(1, 24).all()
        ):

            timestamps = timestamps + pd.to_timedelta(
                hours - 1,
                unit="h"
            )

    target = pd.to_numeric(
        raw[target_col],
        errors="coerce"
    )

    valid = (
        timestamps.notna()
        & target.notna()
        & np.isfinite(target)
    )

    raw = raw.loc[valid].copy()
    timestamps = timestamps.loc[valid]
    target = target.loc[valid]

    order = timestamps.sort_values(
        kind="stable"
    ).index

    raw = raw.loc[order].reset_index(drop=True)

    timestamps = timestamps.loc[order].reset_index(drop=True)

    y = target.loc[order].reset_index(drop=True).astype(float)

    X = pd.DataFrame(index=raw.index)

    excluded = {
        target_col.lower(),
        date_col.lower()
    }

    # Extract numeric predictors.

    for column in raw.columns:

        name = str(column)
        lower_name = name.strip().lower()

        if lower_name in excluded:
            continue

        if any(
            token in lower_name
            for token in LEAKAGE_TOKENS
        ):
            continue

        numeric = pd.to_numeric(
            raw[column],
            errors="coerce"
        )

        original_nonmissing = raw[column].notna().sum()

        if original_nonmissing == 0:
            continue

        numeric_share = (
            numeric.notna().sum()
            / original_nonmissing
        )

        if numeric_share >= 0.90:
            X[name] = numeric

    # Calendar predictors.
    # Updated for pandas 4 compatibility.

    hour = timestamps.dt.hour.astype(float)

    day_of_week = timestamps.dt.day_of_week.astype(float)

    month = timestamps.dt.month.astype(float)

    day_of_year = timestamps.dt.day_of_year.astype(float)

    X["time_hour_sin"] = np.sin(
        2 * np.pi * hour / 24.0
    )

    X["time_hour_cos"] = np.cos(
        2 * np.pi * hour / 24.0
    )

    X["time_dow_sin"] = np.sin(
        2 * np.pi * day_of_week / 7.0
    )

    X["time_dow_cos"] = np.cos(
        2 * np.pi * day_of_week / 7.0
    )

    X["time_month_sin"] = np.sin(
        2 * np.pi * (month - 1) / 12.0
    )

    X["time_month_cos"] = np.cos(
        2 * np.pi * (month - 1) / 12.0
    )

    X["time_year_sin"] = np.sin(
        2 * np.pi * (day_of_year - 1) / 365.25
    )

    X["time_year_cos"] = np.cos(
        2 * np.pi * (day_of_year - 1) / 365.25
    )

    usable = [
        column
        for column in X.columns
        if (
            X[column].notna().any()
            and X[column].nunique(dropna=True) > 1
        )
    ]

    X = X[usable]

    if X.shape[1] == 0:
        raise ValueError(
            "No usable numeric predictors remain after filtering."
        )

    return X, y, timestamps


def prepare_splits(X, y, timestamps):

    n = len(X)

    train_end = int(n * TRAIN_FRACTION)

    validation_end = int(
        n * (TRAIN_FRACTION + VALIDATION_FRACTION)
    )

    if (
        train_end < 100
        or validation_end <= train_end
        or n - validation_end < 50
    ):

        raise ValueError(
            f"Not enough observations for chronological split: n={n}"
        )

    X_train_raw = X.iloc[:train_end].copy()

    X_val_raw = X.iloc[train_end:validation_end].copy()

    X_test_raw = X.iloc[validation_end:].copy()

    y_train = y.iloc[:train_end].copy()

    y_val = y.iloc[train_end:validation_end].copy()

    y_test = y.iloc[validation_end:].copy()

    t_train = timestamps.iloc[:train_end].copy()

    t_val = timestamps.iloc[train_end:validation_end].copy()

    t_test = timestamps.iloc[validation_end:].copy()

    # Learn missing-value treatment from training only.

    missing_rate = X_train_raw.isna().mean()

    columns = [
        column
        for column in X.columns
        if missing_rate[column] <= 0.40
    ]

    X_train_raw = X_train_raw[columns]

    X_val_raw = X_val_raw[columns]

    X_test_raw = X_test_raw[columns]

    medians = X_train_raw.median(
        numeric_only=True
    )

    columns = [
        column
        for column in columns
        if (
            column in medians.index
            and pd.notna(medians[column])
        )
    ]

    if not columns:
        raise ValueError(
            "No predictors remain after missing-value filtering."
        )

    X_train = X_train_raw[columns].fillna(
        medians[columns]
    )

    X_val = X_val_raw[columns].fillna(
        medians[columns]
    )

    X_test = X_test_raw[columns].fillna(
        medians[columns]
    )

    varying = [
        column
        for column in columns
        if X_train[column].nunique(dropna=True) > 1
    ]

    X_train = X_train[varying]

    X_val = X_val[varying]

    X_test = X_test[varying]

    if X_train.shape[1] == 0:
        raise ValueError(
            "All predictors are constant in the training sample."
        )

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        t_train, t_val, t_test,
        varying
    )


def calculate_metrics(y_true, y_pred, spike_threshold):

    actual = np.asarray(y_true, dtype=float)

    predicted = np.asarray(y_pred, dtype=float)

    absolute_error = np.abs(actual - predicted)

    actual_spike = (
        np.abs(actual) >= spike_threshold
    )

    predicted_spike = (
        np.abs(predicted) >= spike_threshold
    )

    spike_mae = (
        float(absolute_error[actual_spike].mean())
        if actual_spike.any()
        else np.nan
    )

    spike_recall = (
        float(
            np.sum(actual_spike & predicted_spike)
            / np.sum(actual_spike)
        )
        if actual_spike.any()
        else np.nan
    )

    spike_precision = (
        float(
            np.sum(actual_spike & predicted_spike)
            / np.sum(predicted_spike)
        )
        if predicted_spike.any()
        else np.nan
    )

    return {
        "model": MODEL_NAME,
        "n": int(len(actual)),

        "mae": float(
            mean_absolute_error(actual, predicted)
        ),

        "rmse": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),

        "r2": float(
            r2_score(actual, predicted)
        ),

        "bias": float(
            np.mean(predicted - actual)
        ),

        "direction_accuracy": float(
            np.mean(
                np.sign(actual) == np.sign(predicted)
            )
        ),

        "spike_threshold_abs_dart": float(spike_threshold),

        "spike_n": int(actual_spike.sum()),

        "spike_mae": spike_mae,

        "spike_recall": spike_recall,

        "spike_precision": spike_precision,
    }


def save_predictions(path, timestamps, y_true, y_pred):

    frame = pd.DataFrame({

        "delivery_timestamp": (
            pd.to_datetime(timestamps).astype(str)
        ),

        "actual_dart": np.asarray(
            y_true,
            dtype=float
        ),

        "predicted_dart": np.asarray(
            y_pred,
            dtype=float
        ),

    })

    frame["error"] = (
        frame["predicted_dart"] - frame["actual_dart"]
    )

    frame["absolute_error"] = frame["error"].abs()

    frame["model"] = MODEL_NAME

    frame.to_csv(path, index=False)


def save_monthly_performance(
    timestamps,
    y_true,
    y_pred,
    path
):

    frame = pd.DataFrame({

        "delivery_timestamp": (
            pd.to_datetime(timestamps).values
        ),

        "actual_dart": np.asarray(
            y_true,
            dtype=float
        ),

        "predicted_dart": np.asarray(
            y_pred,
            dtype=float
        ),

    })

    frame["month"] = (
        frame["delivery_timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    rows = []

    for month, group in frame.groupby(
        "month",
        sort=True
    ):

        rows.append({

            "month": month,

            "n": int(len(group)),

            "mae": float(
                mean_absolute_error(
                    group["actual_dart"],
                    group["predicted_dart"]
                )
            ),

            "rmse": float(
                np.sqrt(
                    mean_squared_error(
                        group["actual_dart"],
                        group["predicted_dart"]
                    )
                )
            ),

            "bias": float(
                (
                    group["predicted_dart"]
                    - group["actual_dart"]
                ).mean()
            ),

        })

    pd.DataFrame(rows).to_csv(
        path,
        index=False
    )


def update_comparison():

    COMPARISON_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    for name in [
        "lightgbm",
        "catboost",
        "xgboost"
    ]:

        path = (
            ROOT
            / "results"
            / name
            / "test_metrics.csv"
        )

        if path.exists():

            try:

                result = pd.read_csv(path)

                if not result.empty:
                    rows.append(result.iloc[0].to_dict())

            except Exception as error:

                print(
                    f"Warning: could not read {path}: {error}"
                )

    if rows:

        comparison = pd.DataFrame(rows)

        if "mae" in comparison.columns:

            comparison = comparison.sort_values(
                "mae",
                ascending=True
            )

        comparison.to_csv(
            COMPARISON_DIR
            / "boosting_models_comparison.csv",
            index=False
        )


def fit_validation_model(
    X_train,
    y_train,
    X_val,
    y_val
):

    model = CatBoostRegressor(

        iterations=3000,

        depth=5,

        learning_rate=0.025,

        loss_function="MAE",

        eval_metric="MAE",

        l2_leaf_reg=10.0,

        random_seed=RANDOM_SEED,

        verbose=False,

        allow_writing_files=False,

        thread_count=-1,

    )

    model.fit(

        X_train,

        y_train,

        eval_set=(X_val, y_val),

        early_stopping_rounds=150,

        use_best_model=True,

        verbose=False,

    )

    best_iteration = model.get_best_iteration()

    if best_iteration is None or best_iteration < 0:
        best_iteration = 999

    return model, max(50, int(best_iteration) + 1)


def fit_final_model(X, y, iterations):

    model = CatBoostRegressor(

        iterations=int(iterations),

        depth=5,

        learning_rate=0.025,

        loss_function="MAE",

        eval_metric="MAE",

        l2_leaf_reg=10.0,

        random_seed=RANDOM_SEED,

        verbose=False,

        allow_writing_files=False,

        thread_count=-1,

    )

    model.fit(X, y, verbose=False)

    return model


def main():

    parser = argparse.ArgumentParser(
        description="Train CatBoost for DART forecasting"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Processed DART feature CSV"
    )

    args = parser.parse_args()

    input_path = resolve_input(args.input)

    X, y, timestamps = load_data(input_path)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        t_train, t_val, t_test,
        feature_names
    ) = prepare_splits(
        X,
        y,
        timestamps
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    validation_model, best_iteration = fit_validation_model(
        X_train,
        y_train,
        X_val,
        y_val
    )

    validation_predictions = validation_model.predict(
        X_val
    )

    # Refit using training and validation observations.

    X_train_validation = pd.concat(
        [X_train, X_val],
        axis=0
    )

    y_train_validation = pd.concat(
        [y_train, y_val],
        axis=0
    )

    final_model = fit_final_model(
        X_train_validation,
        y_train_validation,
        best_iteration
    )

    test_predictions = final_model.predict(
        X_test
    )

    spike_threshold = float(
        np.quantile(
            np.abs(y_train.to_numpy()),
            0.95
        )
    )

    validation_metrics = calculate_metrics(
        y_val,
        validation_predictions,
        spike_threshold
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
        spike_threshold
    )

    save_predictions(
        RESULT_DIR / "validation_predictions.csv",
        t_val,
        y_val,
        validation_predictions
    )

    save_predictions(
        RESULT_DIR / "test_predictions.csv",
        t_test,
        y_test,
        test_predictions
    )

    pd.DataFrame(
        [validation_metrics]
    ).to_csv(
        RESULT_DIR / "validation_metrics.csv",
        index=False
    )

    pd.DataFrame(
        [test_metrics]
    ).to_csv(
        RESULT_DIR / "test_metrics.csv",
        index=False
    )

    save_monthly_performance(
        t_test,
        y_test,
        test_predictions,
        RESULT_DIR / "monthly_test_performance.csv"
    )

    joblib.dump(
        final_model,
        RESULT_DIR / "model.joblib"
    )

    importance = pd.DataFrame({

        "feature": feature_names,

        "importance": np.asarray(
            final_model.feature_importances_,
            dtype=float
        ),

    }).sort_values(
        "importance",
        ascending=False
    )

    importance.to_csv(
        RESULT_DIR / "feature_importance.csv",
        index=False
    )

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

        "validation_end": str(t_val.iloc[-1]),

        "test_start": str(t_test.iloc[0]),

        "test_end": str(t_test.iloc[-1]),

        "test_metrics": test_metrics,

        "split": "chronological 70% train / 15% validation / 15% test",

    }

    with open(
        RESULT_DIR / "run_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            allow_nan=True
        )

    update_comparison()

    print("=" * 72)
    print("CATBOOST DART FORECAST")
    print("=" * 72)

    print(f"Input:       {input_path}")
    print(f"Observations:{len(y):,}")
    print(f"Features:    {len(feature_names)}")
    print(f"Best trees:  {best_iteration}")
    print(f"Test MAE:    {test_metrics['mae']:.4f}")
    print(f"Test RMSE:   {test_metrics['rmse']:.4f}")
    print(f"Test R2:     {test_metrics['r2']:.4f}")
    print(f"Spike MAE:   {test_metrics['spike_mae']:.4f}")
    print(f"Results:     {RESULT_DIR}")

    print("=" * 72)


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print(
            f"ERROR: {error}",
            file=sys.stderr
        )

        raise
