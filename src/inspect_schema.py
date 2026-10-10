
import mlflow
import pandas as pd

MODEL_URI = "models:/bank-fraud-risk@champion"

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load the registered champion model
model = mlflow.pyfunc.load_model(MODEL_URI)

# Retrieve the input schema saved during training
schema = model.metadata.get_input_schema()

print("MODEL INPUT SCHEMA")
print(schema)

# Compare it with the original dataset
df = pd.read_csv("data/Base.csv", nrows=5)

print("\nFEATURE NAMES AND DATA TYPES")

for feature in schema.inputs:
    name = feature.name
    print(
        f"{name:40} "
        f"MLflow: {str(feature.type):12} "
        f"Pandas: {str(df[name].dtype)}"
    )
