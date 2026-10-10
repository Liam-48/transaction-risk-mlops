import pandas as pd
import mlflow
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model_uri = "models:/bank-fraud-risk@champion"

# Load trained pipeline 
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Sample
sample = pd.read_csv("data/Base.csv", nrows=5)
X_sample = sample.drop(columns=["fraud_bool", "month"])

# Make predictions
probabilities = np.asarray(loaded_model.predict(X_sample))

print("Prediction shape:", probabilities.shape)
print("Fraud probabilities:", probabilities[:, 1])
print("Probabilities sum to 1:",
      np.allclose(probabilities.sum(axis=1), 1))
