import mlflow
import numpy as np
import matplotlib.pyplot as plt

from src.data import load_data, split_data
from src.evaluation import evaluate_at_capacity

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model = mlflow.pyfunc.load_model(
    "models:/bank-fraud-risk@champion"
)

df = load_data("data/Base.csv")

X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)

y_proba = np.asarray(model.predict(X_val))[:, 1]

del df, X_train, y_train, X_test, y_test

capacities = np.linspace(0.005, 0.10, 20)

recalls = []

for capacity in capacities:
    result = evaluate_at_capacity(
        y_val, y_proba, capacity
    )

    recalls.append(result["recall"])

recalls = np.array(recalls)
plt.figure(figsize=(9, 5))

plt.plot(
    capacities * 100,
    recalls * 100,
    marker="o",
    label="XGBoost"
)
plt.plot(
    capacities * 100,
    capacities * 100,
    linestyle="--",
    label="Random selection"
)

plt.title("Fraud Detection vs. Investigation Capacity")
plt.xlabel("Applications Reviewed (%)")
plt.ylabel("Fraud Recall (%)")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

from pathlib import Path

Path("reports").mkdir(exist_ok=True)

plt.savefig(
    "reports/review_capacity.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

