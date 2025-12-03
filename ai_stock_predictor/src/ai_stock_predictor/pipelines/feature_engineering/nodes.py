import pandas as pd
import numpy as np


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure sorted BEFORE computing anything
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    df['Typical price'] = (df['High'] + df['Low'] + df['Close']) / 3

    df["MA_10"] = df.groupby("Symbol")["Close"].transform(lambda x: x.rolling(10).mean())
    df["MA_50"] = df.groupby("Symbol")["Close"].transform(lambda x: x.rolling(50).mean())

    df["RSI"] = compute_rsi_grouped(df)

    return df


def add_lag_features(df: pd.DataFrame, lag_days: list[int]) -> pd.DataFrame:
    df = df.copy()

    # Ensure sorted BEFORE shifting
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    for lag in lag_days:
        df[f"lag_{lag}"] = df.groupby("Symbol")["Close"].shift(lag)

    if 'RSI' in df.columns:
        for lag in [1, 2]:
            df[f"RSI_lag_{lag}"] = df.groupby("Symbol")["RSI"].shift(lag)

    if 'MA_10' in df.columns:
        for lag in [1, 2]:
            df[f"MA_10_lag_{lag}"] = df.groupby("Symbol")["MA_10"].shift(lag)

    if 'MA_50' in df.columns:
        for lag in [1, 2]:
            df[f"MA_50_lag_{lag}"] = df.groupby("Symbol")["MA_50"].shift(lag)

    return df

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    df['Date'] = pd.to_datetime(df['Date'])

    df['day_of_week'] = df['Date'].dt.dayofweek       # 0=Monday
    df['day_of_month'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)

    return df

def add_company_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # Ensure datetime types
    df['Date'] = pd.to_datetime(df['Date'])
    df['Founded'] = pd.to_datetime(df['Founded'], errors='coerce')
    df['Date added'] = pd.to_datetime(df['Date added'], errors='coerce')

    # Company age at each row
    df['company_age'] = (df['Date'] - df['Founded']).dt.days / 365.25

    # Years since added to index (e.g. S&P500)
    df['years_since_added'] = df['Date'].dt.year - df['Date added'].dt.year

    # Handle cases where founding date or date added is missing
    df['company_age'] = df['company_age'].fillna(-1)
    df['years_since_added'] = df['years_since_added'].fillna(-1)

    return df

def add_corporate_action_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    df['dividend_lag_1'] = df.groupby("Symbol")["Dividends"].shift(1)
    df['split_lag_1'] = df.groupby("Symbol")["Stock Splits"].shift(1)
    return df

def compute_rsi_grouped(df: pd.DataFrame, window: int = 14) -> pd.Series:
    def compute_rsi(series: pd.Series) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()

        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df = df.sort_values(["Symbol", "Date"])
    return df.groupby("Symbol")["Close"].transform(compute_rsi)
