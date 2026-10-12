
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.schemas import ApplicationRequest


DATA_PATH = Path("data/Base.csv")
REPORT_DIR = Path("reports")

# Reuse the same feature definitions as the API
FIELDS = ApplicationRequest.model_fields
FEATURES = list(FIELDS)

CATEGORICAL = [
    name for name, field in FIELDS.items()
    if field.annotation is str
]

NUMERICAL = [
    name for name, field in FIELDS.items()
    if field.annotation in (int, float)
]

# Missing-value conventions from our preprocessing
NEGATIVE_ONE_MISSING = {
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
}



def load_monitoring_data():
    df = pd.read_csv(
        DATA_PATH,
        usecols=["month", *FEATURES]
    )

    reference = df.loc[
        df["month"].between(0, 5), FEATURES
    ].copy()

    current = df.loc[
        df["month"] == 6, FEATURES
    ].copy()

    return reference, current


def missing_mask(series, feature):
    mask = series.isna()

    if feature in NEGATIVE_ONE_MISSING:
        mask = mask | series.eq(-1)

    if feature == "intended_balcon_amount":
        mask = mask | series.lt(0)

    return mask


def categorical_distance(reference, current):
    # Total Variation Distance between distributions
    p = reference.value_counts(normalize=True)
    q = current.value_counts(normalize=True)

    categories = p.index.union(q.index)

    p = p.reindex(categories, fill_value=0)
    q = q.reindex(categories, fill_value=0)

    return float(0.5 * (p - q).abs().sum())


def monitor_features(reference, current):
    results = []

    for feature in FEATURES:
        ref = reference[feature]
        cur = current[feature]

        # Identify missing / sentinel values
        ref_missing = missing_mask(ref, feature)
        cur_missing = missing_mask(cur, feature)

        ref_valid = ref[~ref_missing]
        cur_valid = cur[~cur_missing]

        if ref_valid.empty or cur_valid.empty:
            raise ValueError(
                f"No valid observations for {feature}"
            )

        if feature in NUMERICAL:
            drift = float(
                ks_2samp(ref_valid, cur_valid).statistic
            )
            metric = "KS"

        else:
            drift = categorical_distance(
                ref_valid, cur_valid
            )
            metric = "TVD"

        ref_missing_pct = 100 * ref_missing.mean()
        cur_missing_pct = 100 * cur_missing.mean()

        results.append({
            "feature": feature,
            "type": "numerical"
                if feature in NUMERICAL else "categorical",
            "metric": metric,
            "drift": drift,
            "reference_missing_pct": ref_missing_pct,
            "current_missing_pct": cur_missing_pct,
            "missing_delta_pp": (
                cur_missing_pct - ref_missing_pct
            ),
        })

    return pd.DataFrame(results)


def predict_in_batches(model, X, batch_size=25_000):
    scores = []

    for start in range(0, len(X), batch_size):
        batch = X.iloc[start:start + batch_size]

        probabilities = np.asarray(
            model.predict(batch)
        )

        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError(
                "Expected probabilities for two classes"
            )

        scores.append(probabilities[:, 1])

    return np.concatenate(scores)


def monitor_predictions(reference, current):
    import mlflow

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000"
    )

    model_uri = os.getenv(
        "MODEL_URI",
        "models:/bank-fraud-risk@champion"
    )

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    registered = client.get_model_version_by_alias(
        "bank-fraud-risk",
        "champion"
    )

    model_version = registered.version
    resolved_uri = f"models:/bank-fraud-risk/{model_version}"

    model = mlflow.pyfunc.load_model(resolved_uri)

    ref_scores = predict_in_batches(
        model, reference[FEATURES]
    )

    cur_scores = predict_in_batches(
        model, current[FEATURES]
    )

    return {
        "model": model_uri,
        "reference_count": len(ref_scores),
        "current_count": len(cur_scores),
        "reference_mean_score": float(ref_scores.mean()),
        "current_mean_score": float(cur_scores.mean()),
        "reference_p95_score": float(np.quantile(ref_scores, 0.95)),
        "current_p95_score": float(np.quantile(cur_scores, 0.95)),
        "prediction_ks": float(
            ks_2samp(ref_scores, cur_scores).statistic
        ),
        "model_version": str(model_version),
        "resolved_model_uri": resolved_uri,
    }


def save_drift_chart(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = (
        report[report["metric"] == "KS"]
        .nlargest(10, "drift")
        .sort_values("drift")
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(top["feature"], top["drift"])

    ax.set_title(
        "Numerical Feature Drift\n"
        "Historical reference vs. monitoring batch",
        fontsize=15,
        fontweight="bold",
        pad=16
    )

    ax.set_xlabel("Kolmogorov–Smirnov distance")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)

    xmax = max(top["drift"].max(), 0.05)

    ax.set_xlim(0, xmax * 1.2)

    for i, value in enumerate(top["drift"]):
        ax.text(
            value + xmax * 0.015,
            i,
            f"{value:.3f}",
            va="center",
            fontsize=9
        )

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR / "feature_drift.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


def log_monitoring_run(feature_report, prediction_report,
                       reference_count, current_count):
    import mlflow

    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000"
        )
    )

    mlflow.set_experiment("bank-fraud-monitoring")

    numerical = feature_report[
        feature_report["metric"] == "KS"
    ]

    categorical = feature_report[
        feature_report["metric"] == "TVD"
    ]

    with mlflow.start_run(run_name="monitor_month_6") as run:
        mlflow.log_params({
            "reference_months": "0-5",
            "current_month": "6",
            "reference_count": reference_count,
            "current_count": current_count,
            "model_version": prediction_report["model_version"],
            "model_uri": prediction_report["resolved_model_uri"],
        })

        mlflow.log_metrics({
            "max_numerical_ks": float(numerical["drift"].max()),
            "max_categorical_tvd": float(categorical["drift"].max()),
            "prediction_ks": prediction_report["prediction_ks"],
            "reference_mean_score": prediction_report["reference_mean_score"],
            "current_mean_score": prediction_report["current_mean_score"],
            "reference_p95_score": prediction_report["reference_p95_score"],
            "current_p95_score": prediction_report["current_p95_score"],
        })

        # Attach reports for reproducibility
        files = [
            "monitoring_features.csv",
            "monitoring_predictions.json",
            "feature_drift.png",
        ]

        for filename in files:
            path = REPORT_DIR / filename

            if path.is_file():
                mlflow.log_artifact(
                    str(path),
                    artifact_path="monitoring"
                )
            else:
                raise FileNotFoundError(path)

        print(f"\nMonitoring run logged: {run.info.run_id}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--with-scores",
        action="store_true",
        help="Also measure model prediction drift"
    )

    args = parser.parse_args()

    REPORT_DIR.mkdir(exist_ok=True)

    reference, current = load_monitoring_data()

    print(f"Reference applications: {len(reference):,}")
    print(f"Current applications: {len(current):,}")

    report = monitor_features(reference, current)

    report.to_csv(
        REPORT_DIR / "monitoring_features.csv",
        index=False
    )

    save_drift_chart(report)

    print("\nTop numerical drift:")
    print(
        report[report["type"] == "numerical"]
        .sort_values("drift", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\nCategorical drift:")
    print(
        report[report["type"] == "categorical"]
        .sort_values("drift", ascending=False)
        .to_string(index=False)
    )

    if args.with_scores:
        prediction_report = monitor_predictions(
            reference, current
        )

        output = REPORT_DIR / "monitoring_predictions.json"

        output.write_text(
            json.dumps(prediction_report, indent=2),
            encoding="utf-8"
        )

        log_monitoring_run(
            report,
            prediction_report,
            len(reference),
            len(current)
        )

        print("\nPrediction drift:")
        print(json.dumps(prediction_report, indent=2))


if __name__ == "__main__":
    main()
