import pandas as pd
import numpy as np

def add_lag_features(df: pd.DataFrame, lag_days: list[int]) -> pd.DataFrame:
    df = df.copy()
    for lag in lag_days:
        df[f"lag_{lag}"] = df["Close"].shift(lag)
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['Typical price'] = (df['High'] + df['Low'] + df['Close']) / 3

    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["RSI"] = compute_rsi(df["Close"], window=14)

    return df
    
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi