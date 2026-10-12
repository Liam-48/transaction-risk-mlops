
import argparse

import mlflow
import numpy as np
import pandas as pd


def verify_model(model_uri: str):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    model = mlflow.pyfunc.load_model(model_uri)

    sample = pd.read_csv("data/Base.csv", nrows=5)
    X = sample.drop(columns=["fraud_bool", "month"])

    probabilities = np.asarray(model.predict(X))

    # Verify the predict_proba interface
    if probabilities.shape != (len(X), 2):
        raise ValueError(
            f"Unexpected prediction shape: {probabilities.shape}"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions contain non-finite values")

    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Predictions outside [0, 1]")

    if not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Class probabilities do not sum to 1")

    print(f"Verified model: {model_uri}")
    print("Prediction shape:", probabilities.shape)
    print("Fraud probabilities:", probabilities[:, 1])
    print("All validation checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-uri",
        default="models:/bank-fraud-risk@champion",
    )
    args = parser.parse_args()

    verify_model(args.model_uri)
