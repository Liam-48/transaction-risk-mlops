
import argparse
import os

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.metrics import average_precision_score

from src.data import load_data, split_data
from src.evaluation import evaluate_at_capacity
from src.modeling import (
    build_xgb_pipeline,
    build_logistic_pipeline,
)


XGB_BASELINE_PARAMS = {
    "n_estimators": 150,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "n_jobs": 2,
    "random_state": 0,
    "eval_metric": "logloss",
}


def main(model_type):
    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000",
        )
    )
    mlflow.set_experiment("bank-fraud-detection")

    df = load_data("data/Base.csv")
    X_train, y_train, X_val, y_val, _, _ = split_data(df)

    if model_type == "logistic":
        model = build_logistic_pipeline(X_train)
        params = {"max_iter": 1000}
        model_name = "LogisticRegression"
    else:
        model = build_xgb_pipeline(
            X_train, XGB_BASELINE_PARAMS
        )
        params = XGB_BASELINE_PARAMS
        model_name = "XGBClassifier"

    with mlflow.start_run(
        run_name=f"{model_type}_baseline"
    ):
        model.fit(X_train, y_train)

        scores = model.predict_proba(X_val)[:, 1]

        ap = average_precision_score(y_val, scores)
        results = evaluate_at_capacity(
            y_val, scores, capacity=0.01
        )

        mlflow.log_params({
            "model": model_name,
            "training_samples": len(X_train),
            "training_months": "0-5",
            "validation_month": 6,
            "review_capacity": 0.01,
            **params,
        })

        mlflow.log_metrics({
            "validation_ap": ap,
            "precision_at_1pct": results["precision"],
            "recall_at_1pct": results["recall"],
            "fraud_captured_at_1pct": results["fraud_captured"],
        })

        example = X_val.head(5)

        signature = infer_signature(
            example,
            model.predict_proba(example),
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="fraud_pipeline",
            signature=signature,
            input_example=example,
            code_paths=["src"],
            pyfunc_predict_fn="predict_proba",
            serialization_format="cloudpickle",
        )

        print(f"\nModel: {model_name}")
        print(f"Validation AP: {ap:.5f}")
        print(f"Precision@1%: {results['precision']:.4f}")
        print(f"Recall@1%: {results['recall']:.4f}")
        print(f"Frauds captured: {results['fraud_captured']}")
        print(f"Applications reviewed: {results['flagged']}")
        print(f"MLflow model: {model_info.model_uri}")

        print("Champion alias unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["logistic", "xgboost"],
        required=True,
    )

    args = parser.parse_args()
    main(args.model)
