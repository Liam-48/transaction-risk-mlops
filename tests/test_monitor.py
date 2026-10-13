
import numpy as np
import pandas as pd
import pytest

from src.monitor import (
    CATEGORICAL,
    FEATURES,
    categorical_distance,
    missing_mask,
    monitor_features,
    predict_in_batches,
)


def test_categorical_distance():
    reference = pd.Series(["A", "A", "B", "B"])
    current = pd.Series(["A", "B", "B", "B"])

    distance = categorical_distance(reference, current)

    assert distance == pytest.approx(0.25)


def test_missing_sentinels():
    addresses = pd.Series([-1, 10, -1, 24])
    balance = pd.Series([-50.0, 0.0, 100.0])

    assert missing_mask(
        addresses, "prev_address_months_count"
    ).sum() == 2

    assert missing_mask(
        balance, "intended_balcon_amount"
    ).sum() == 1


def test_feature_monitoring():
    # Construct valid data for all 30 model features
    reference = pd.DataFrame({
        name: ["A"] * 10 if name in CATEGORICAL else [0] * 10
        for name in FEATURES
    })

    current = reference.copy()

    # Deliberately introduce complete distribution drift
    current["velocity_24h"] = 1.0
    current["housing_status"] = "B"

    # Introduce a 20 percentage-point increase in missingness
    current.loc[:1, "prev_address_months_count"] = -1

    report = monitor_features(reference, current)
    report = report.set_index("feature")

    assert len(report) == 30

    assert report.loc["velocity_24h", "drift"] == pytest.approx(1.0)
    assert report.loc["housing_status", "drift"] == pytest.approx(1.0)

    assert report.loc[
        "prev_address_months_count", "missing_delta_pp"
    ] == pytest.approx(20.0)


def test_batched_predictions():
    class FakeModel:
        def predict(self, batch):
            p = batch["value"].to_numpy() / 10
            return np.column_stack([1 - p, p])

    X = pd.DataFrame({"value": [1, 2, 3, 4, 5]})

    scores = predict_in_batches(
        FakeModel(), X, batch_size=2
    )

    assert len(scores) == 5
    np.testing.assert_allclose(
        scores,
        [0.1, 0.2, 0.3, 0.4, 0.5]
    )

def test_monitoring_policy_flags_large_drift():
    import pandas as pd
    from src.monitor import assess_drift

    report = pd.DataFrame([
        {
            "feature": "velocity_4w",
            "metric": "KS",
            "drift": 0.69,
            "missing_delta_pp": 0.0,
        },
        {
            "feature": "income",
            "metric": "KS",
            "drift": 0.05,
            "missing_delta_pp": 0.0,
        },
    ])

    decision = assess_drift(report)

    assert decision["status"] == "review_recommended"
    assert len(decision["findings"]) == 1
    assert decision["findings"][0]["feature"] == "velocity_4w"
