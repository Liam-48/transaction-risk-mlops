
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.preprocessing import negative_to_nan


def build_preprocessor(X_train):
    sentinel_cols = [
        "prev_address_months_count",
        "current_address_months_count",
        "bank_months_count",
        "session_length_in_minutes",
    ]

    categorical_cols = [
        "payment_type",
        "employment_status",
        "housing_status",
        "source",
        "device_os",
    ]

    special_cols = sentinel_cols + [
        "intended_balcon_amount"
    ]

    numeric_cols = [
        col for col in X_train.select_dtypes(
            include="number"
        ).columns
        if col not in special_cols
    ]

    sentinel_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            missing_values=-1,
            strategy="median",
            add_indicator=True,
        )),
        ("scaler", StandardScaler()),
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    balance_pipeline = Pipeline([
        ("negative_to_nan",
         FunctionTransformer(negative_to_nan)),
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True,
        )),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer([
        ("sentinel", sentinel_pipeline, sentinel_cols),
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_cols),
        ("intended_balcon_amount",
         balance_pipeline,
         ["intended_balcon_amount"]),
    ])


def build_xgb_pipeline(X_train, params):
    return Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("classifier", XGBClassifier(**params)),
    ])


def build_logistic_pipeline(X_train):
    return Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
