
from contextlib import asynccontextmanager

import mlflow
import numpy as np
import pandas as pd

from fastapi import FastAPI, Request
from src.schemas import ApplicationRequest, PredictionResponse

import os

MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/bank-fraud-risk@champion"
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}", flush=True)

    app.state.model = mlflow.pyfunc.load_model(MODEL_URI)
    yield


app = FastAPI(
    title="Bank Fraud Risk API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_URI
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(application: ApplicationRequest, request: Request):

    # 1. Pydantic object -> dictionary
    data = application.model_dump()

    # 2. Dictionary -> one-row DataFrame
    X = pd.DataFrame([data])

    # 3. Load champion model
    model = request.app.state.model

    # 4. Generate class probabilities
    probabilities = np.asarray(model.predict(X))

    # 5. Extract probability of fraud (class 1)
    fraud_probability = float(probabilities[0, 1])

    return {
        "fraud_probability": fraud_probability,
        "model": MODEL_URI
    }
