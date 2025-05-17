import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pickle

def prepare_model_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.dropna(inplace=True)
    df["target"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    return df

def split_data(df: pd.DataFrame, test_size: float, random_state: int):
    X = df.drop(columns=["Date", "target"])
    y = df["target"]

    # Convert all features to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Optional: drop rows where features or target are NaN
    mask = X.notnull().all(axis=1) & y.notnull()
    X = X[mask]
    y = y[mask]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, model_params: dict) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**model_params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test) -> dict:
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return {"mse": mse}

def make_predictions(model, X_test) -> pd.DataFrame:
    preds = model.predict(X_test)
    return pd.DataFrame({"prediction": preds})