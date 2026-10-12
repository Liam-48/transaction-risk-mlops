
import math

import numpy as np
from sklearn.metrics import average_precision_score

import argparse
import json
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient


def evaluate_candidate(y_true, scores, capacity=0.01):
    y = np.asarray(y_true)
    scores = np.asarray(scores)

    if y.ndim != 1 or scores.ndim != 1:
        raise ValueError("Expected one-dimensional arrays")

    if len(y) != len(scores) or len(y) == 0:
        raise ValueError("Invalid array lengths")

    if not 0 < capacity <= 1:
        raise ValueError("Capacity must be in (0, 1]")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary")

    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")

    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("Both classes must be present")

    # Number of applications analysts can review
    k = math.ceil(len(y) * capacity)

    # Highest risk first; stable ordering for ties
    top_indices = np.argsort(-scores, kind="stable")[:k]

    frauds_captured = int(y[top_indices].sum())
    total_frauds = int(y.sum())

    return {
        "average_precision": float(
            average_precision_score(y, scores)
        ),
        "reviewed": k,
        "frauds_captured": frauds_captured,
        "precision_at_capacity": frauds_captured / k,
        "recall_at_capacity": frauds_captured / total_frauds,
    }


def decide_promotion(
    y_true,
    champion_scores,
    challenger_scores,
    capacity=0.01,
    min_extra_frauds=1,
):
    champion = evaluate_candidate(
        y_true, champion_scores, capacity
    )

    challenger = evaluate_candidate(
        y_true, challenger_scores, capacity
    )

    fraud_gain = (
        challenger["frauds_captured"]
        - champion["frauds_captured"]
    )

    # Conservative initial business policy
    capture_improved = fraud_gain >= min_extra_frauds

    ap_not_worse = (
        challenger["average_precision"]
        >= champion["average_precision"]
    )

    eligible = capture_improved and ap_not_worse

    return {
        "eligible_for_promotion": bool(eligible),
        "frauds_gained": fraud_gain,
        "champion": champion,
        "challenger": challenger,
        "checks": {
            "capture_improved": bool(capture_improved),
            "ap_not_worse": bool(ap_not_worse),
        },
    }


def promote_from_report(report_path, apply=False):
    report = json.loads(Path(report_path).read_text())

    decision = report["decision"]
    expected_champion = str(report["champion_version"])
    challenger = str(report["challenger_version"])

    # Verify the saved business decision
    fraud_gain = (
        decision["challenger"]["frauds_captured"]
        - decision["champion"]["frauds_captured"]
    )

    valid_decision = (
        decision["eligible_for_promotion"]
        and all(decision["checks"].values())
        and fraud_gain == decision["frauds_gained"]
        and fraud_gain >= report["min_extra_frauds"]
        and (
            decision["challenger"]["average_precision"]
            >= decision["champion"]["average_precision"]
        )
    )

    if not valid_decision:
        raise RuntimeError("Promotion policy not satisfied")

    if challenger == expected_champion:
        raise RuntimeError("Challenger is already champion")

    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000"
        )
    )

    client = MlflowClient()
    model_name = "bank-fraud-risk"

    # Guard against a stale decision report
    current = client.get_model_version_by_alias(
        model_name, "champion"
    )

    if str(current.version) != expected_champion:
        raise RuntimeError(
            f"Champion changed: expected {expected_champion}, "
            f"found {current.version}"
        )

    # Confirm target version exists in MLflow
    client.get_model_version(model_name, challenger)

    print(f"Current champion: v{current.version}")
    print(f"Candidate: v{challenger}")
    print(f"Additional frauds captured: {fraud_gain}")

    if not apply:
        print("DRY RUN PASSED — no changes made.")
        return

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=challenger,
    )

    updated = client.get_model_version_by_alias(
        model_name, "champion"
    )

    if str(updated.version) != challenger:
        raise RuntimeError("Alias verification failed")

    print(f"PROMOTED — champion now points to v{updated.version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--report",
        default="reports/promotion_decision.json"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually update the MLflow champion alias"
    )

    args = parser.parse_args()

    promote_from_report(args.report, apply=args.apply)