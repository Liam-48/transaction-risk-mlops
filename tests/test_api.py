
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import app


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "application.json"
)


@pytest.fixture
def client(monkeypatch):
    # Create a mock model
    fake_model = Mock()
    fake_model.predict.return_value = np.array([[0.9, 0.1]])

    # Inject the model directly into FastAPI
    monkeypatch.setattr(
        app.state,
        "model",
        fake_model,
        raising=False
    )

    # Do not execute the lifespan startup
    return TestClient(app), fake_model


def test_predict_valid_application(client):
    test_client, fake_model = client

    payload = json.loads(
        EXAMPLE_PATH.read_text(encoding="utf-8")
    )

    response = test_client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["fraud_probability"] == pytest.approx(0.1)

    fake_model.predict.assert_called_once()

    X = fake_model.predict.call_args.args[0]

    assert X.shape == (1, 30)
    assert list(X.columns) == list(payload.keys())


def test_predict_missing_features(client):
    test_client, fake_model = client

    response = test_client.post("/predict", json={})

    assert response.status_code == 422
    fake_model.predict.assert_not_called()


def test_predict_extra_feature(client):
    test_client, fake_model = client

    payload = json.loads(
        EXAMPLE_PATH.read_text(encoding="utf-8")
    )
    payload["unknown_feature"] = 123

    response = test_client.post("/predict", json=payload)

    assert response.status_code == 422
    fake_model.predict.assert_not_called()
