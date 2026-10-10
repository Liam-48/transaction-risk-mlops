import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model_uri = "models:/m-7ac4635a247441b5a8bb2d475d238e96"

model_name = "bank-fraud-risk"

registered = mlflow.register_model(model_uri, model_name)

print("Registered version:", registered.version)

client = MlflowClient()

client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=registered.version
)