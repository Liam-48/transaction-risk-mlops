
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature

from src.data import load_data, split_data
from src.modeling import build_xgb_pipeline
from src.promotion import decide_promotion


MODEL_NAME = "bank-fraud-risk"
CAPACITY = 0.01
MIN_EXTRA_FRAUDS = 5

CHALLENGER_PARAMS = {
    "n_estimators": 250,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "n_jobs": 2,
    "random_state": 0,
    "eval_metric": "logloss",
}


def main():
    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000"
        )
    )
    mlflow.set_experiment("bank-fraud-detection")

    # Freeze the champion version before evaluation
    client = mlflow.MlflowClient()
    champion_version = client.get_model_version_by_alias(
        MODEL_NAME, "champion"
    ).version

    champion_uri = (
        f"models:/{MODEL_NAME}/{champion_version}"
    )

    # Use training and validation periods only
    df = load_data("data/Base.csv")
    df = df.loc[df["month"] <= 6].copy()

    X_train, y_train, X_val, y_val, _, _ = split_data(df)

    # Train candidate with shared preprocessing
    challenger = build_xgb_pipeline(
        X_train, CHALLENGER_PARAMS
    )

    with mlflow.start_run(
        run_name="xgboost_challenger"
    ):
        challenger.fit(X_train, y_train)

        challenger_scores = challenger.predict_proba(
            X_val
        )[:, 1]

        # Load the existing champion, without retraining
        champion = mlflow.pyfunc.load_model(
            champion_uri
        )

        champion_scores = np.asarray(
            champion.predict(X_val)
        )[:, 1]

        # Compare both models on the same validation data
        decision = decide_promotion(
            y_val,
            champion_scores,
            challenger_scores,
            capacity=CAPACITY,
            min_extra_frauds=MIN_EXTRA_FRAUDS,
        )

        # Log the candidate model
        example = X_val.head(5)

        signature = infer_signature(
            example,
            challenger.predict_proba(example)
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=challenger,
            name="fraud_pipeline",
            signature=signature,
            input_example=example,
            code_paths=["src"],
            pyfunc_predict_fn="predict_proba",
            serialization_format="cloudpickle",
        )

        # Register a new version, without changing the alias
        registered = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=MODEL_NAME,
        )

        mlflow.log_params(CHALLENGER_PARAMS)
        mlflow.log_params({
            "training_months": "0-5",
            "validation_month": 6,
            "review_capacity": CAPACITY,
            "minimum_extra_frauds": MIN_EXTRA_FRAUDS,
            "champion_version": champion_version,
        })

        champion_metrics = decision["champion"]
        challenger_metrics = decision["challenger"]

        mlflow.log_metrics({
            "champion_validation_ap":
                champion_metrics["average_precision"],
            "challenger_validation_ap":
                challenger_metrics["average_precision"],
            "champion_frauds_at_1pct":
                champion_metrics["frauds_captured"],
            "challenger_frauds_at_1pct":
                challenger_metrics["frauds_captured"],
            "frauds_gained": decision["frauds_gained"],
            "eligible_for_promotion": int(
                decision["eligible_for_promotion"]
            ),
        })

        report = {
            "champion_version": str(champion_version),
            "challenger_version": str(registered.version),
            "review_capacity": CAPACITY,
            "min_extra_frauds": MIN_EXTRA_FRAUDS,
            "decision": decision,
        }

        output = Path("reports/promotion_decision.json")
        output.parent.mkdir(exist_ok=True, parents=True)
        output.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8"
        )

        mlflow.log_artifact(
            str(output),
            artifact_path="promotion"
        )

    print("\nCHAMPION VS CHALLENGER")
    print(f"Champion version: {champion_version}")
    print(f"Challenger version: {registered.version}")
    print(
        "Champion AP:",
        round(champion_metrics["average_precision"], 5)
    )
    print(
        "Challenger AP:",
        round(challenger_metrics["average_precision"], 5)
    )
    print(
        "Frauds gained:",
        decision["frauds_gained"]
    )
    print(
        "Eligible for promotion:",
        decision["eligible_for_promotion"]
    )
    print("Champion alias unchanged.")


if __name__ == "__main__":
    main()
