from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from data import load_data, split_data
from sklearn.preprocessing import FunctionTransformer
from preprocessing import negative_to_nan
from sklearn.metrics import average_precision_score 
import mlflow
from evaluation import evaluate_at_capacity
from xgboost import XGBClassifier
from sklearn.base import clone
import mlflow.sklearn
from mlflow.models import infer_signature

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("bank-fraud-detection")

df = load_data("data/Base.csv")
X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)

sentinel_cols = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes"
]

special_cols = sentinel_cols + ["intended_balcon_amount"]

categorical_cols = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os"
]

numeric_cols = X_train.select_dtypes(
    include="number"
).columns.tolist()

numeric_cols = [
    col for col in numeric_cols
    if col not in special_cols
]

sentinel_pipeline = Pipeline([
    ("imputer", SimpleImputer(missing_values=-1, strategy="median", add_indicator=True)),
    ("scaler", StandardScaler())
])

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = OneHotEncoder(handle_unknown="ignore")

intended_balcon_amount_pipeline = Pipeline([
    ("negative_to_nan", FunctionTransformer(negative_to_nan)),
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([  
    ("sentinel", sentinel_pipeline, sentinel_cols),
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols),
    ("intended_balcon_amount", intended_balcon_amount_pipeline, ["intended_balcon_amount"])
])

# Define the models pipelines

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

xgb_model = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("classifier", XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method = "hist",
        n_jobs=2,
        random_state=0,
        eval_metric="logloss"
    ))
])

# Train and evaluate the model


""" with mlflow.start_run(run_name="logistic_regression_baseline"):
    # 1. Train the model
    model.fit(X_train, y_train)

    # 2. Evaluate the model on the validation set
    y_proba = model.predict_proba(X_val)[:, 1]

    # 3. Evaluate 
    ap = average_precision_score(y_val, y_proba)

    # 4. Evaluate at capacity
    results = evaluate_at_capacity(y_val, y_proba, capacity=0.01)

    # 5. Log parameters and metrics to MLflow
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("training_samples", len(X_train))
    mlflow.log_param("training_months", "0-5")
    mlflow.log_param("review_capacity", 0.01)

    mlflow.log_metrics({
        "validation_ap": ap,
        "precision_at_1pct": results["precision"],
        "recall_at_1pct": results["recall"],
        "fraud_captured_at_1pct": results["fraud_captured"],
    })  

    # 6. Display results
    print("Logistic Regression Model Evaluation:")
    print(f"Validation AP: {ap:.4f}")
    print(f"Precision@1%: {results['precision']:.4f}")
    print(f"Recall@1%: {results['recall']:.4f}")
    print(f"Fraud captured: {results['fraud_captured']}")
    print(f"Applications flagged: {results['flagged']}") """

with mlflow.start_run(run_name="xgboost_baseline"):
    # 1. Train the model
    xgb_model.fit(X_train, y_train)

    # 2. Evaluate the model on the validation set
    y_proba = xgb_model.predict_proba(X_val)[:, 1]

    # 3. Evaluate 
    ap = average_precision_score(y_val, y_proba)

    # 4. Evaluate at capacity
    results = evaluate_at_capacity(y_val, y_proba, capacity=0.01)

    # 5. Log the model to MLflow
    example = X_val.head(5)

    signature = infer_signature(
        example,
        xgb_model.predict_proba(example)
    )

    model_info = mlflow.sklearn.log_model(
        sk_model=xgb_model,
        name="fraud_pipeline",
        signature=signature,
        input_example=example,
        code_paths=["src/preprocessing.py"],
        pyfunc_predict_fn="predict_proba",
        serialization_format="cloudpickle"
    )

    print("Logged model:", model_info.model_uri)    

    # Log parameters and metrics to MLflow
    mlflow.log_param("model", "XGBClassifier")
    mlflow.log_param("n_estimators", 150)
    mlflow.log_param("max_depth", 4)
    mlflow.log_param("learning_rate", 0.08)
    mlflow.log_param("subsample", 0.8)
    mlflow.log_param("colsample_bytree", 0.8)
    mlflow.log_param("tree_method", "hist")
    mlflow.log_param("n_jobs", 2)
    mlflow.log_param("random_state", 0)
    mlflow.log_param("eval_metric", "logloss")
    mlflow.log_param("training_samples", len(X_train))
    mlflow.log_param("training_months", "0-5")
    mlflow.log_param("review_capacity", 0.01)

    mlflow.log_metrics({
        "validation_ap": ap,
        "precision_at_1pct": results["precision"],
        "recall_at_1pct": results["recall"],
        "fraud_captured_at_1pct": results["fraud_captured"],
    })

    # 6. Display results
    print("XGBoost Model Evaluation:")
    print(f"Validation AP: {ap:.4f}")
    print(f"Precision@1%: {results['precision']:.4f}")
    print(f"Recall@1%: {results['recall']:.4f}")
    print(f"Fraud captured: {results['fraud_captured']}")
    print(f"Applications flagged: {results['flagged']}")