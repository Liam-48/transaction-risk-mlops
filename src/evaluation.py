import numpy as np

def evaluate_at_capacity(y_true, y_proba, capacity=0.01):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    n = len(y_true)
    k = max(1, int(np.ceil(n * capacity)))

    top_indices = np.argsort(y_proba)[-k:]

    fraud_captured = y_true[top_indices].sum()
    total_fraud = y_true.sum()

    precision = fraud_captured / k
    recall = fraud_captured / total_fraud if total_fraud > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "flagged": k,
        "fraud_captured": int(fraud_captured)
    }