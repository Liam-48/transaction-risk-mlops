import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data

def split_data(df):
    train = df[df["month"] <= 5]   # training set
    val = df[df["month"] == 6]     # validation set
    test = df[df["month"] == 7]    # test set

    # Split features and target variable
    X_train = train.drop(columns=["fraud_bool", "month"])
    y_train = train["fraud_bool"]

    X_val = val.drop(columns=["fraud_bool", "month"])
    y_val = val["fraud_bool"]

    X_test = test.drop(columns=["fraud_bool", "month"])
    y_test = test["fraud_bool"]    

    return X_train, y_train, X_val, y_val, X_test, y_test

