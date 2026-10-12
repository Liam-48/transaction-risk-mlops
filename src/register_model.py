
import argparse
import os

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


MODEL_NAME = "bank-fraud-risk"


def get_champion(client):
    try:
        return client.get_model_version_by_alias(
            MODEL_NAME, "champion"
        )
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        return None


def main(model_uri, apply=False):
    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000",
        )
    )

    client = MlflowClient()

    current = get_champion(client)

    if current is not None:
        raise RuntimeError(
            f"Champion already exists: v{current.version}. "
            "Initialization refused."
        )

    if not apply:
        print(f"Candidate artifact: {model_uri}")
        print("DRY RUN — no changes made.")
        return

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    # Recheck before assigning the initial alias.
    if get_champion(client) is not None:
        raise RuntimeError(
            "Champion appeared during registration. "
            "Alias unchanged."
        )

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=registered.version,
    )

    print(f"Initial champion: v{registered.version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--apply", action="store_true")

    args = parser.parse_args()
    main(args.model_uri, args.apply)
