import yfinance as yf
import pandas as pd

def download_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    return df

def clean_stock_data(df: pd.DataFrame, fillna_method: str = "ffill") -> pd.DataFrame:
    if fillna_method == "ffill":
        df.fillna(method="ffill", inplace=True)
    elif fillna_method == "bfill":
        df.fillna(method="bfill", inplace=True)
    return df