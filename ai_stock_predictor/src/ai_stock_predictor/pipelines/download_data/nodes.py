import yfinance as yf
import pandas as pd
from tqdm import tqdm
import time
from pathlib import Path
import os
import requests

def download_stocks_information() -> pd.DataFrame:
    p = os.getcwd() + '/data/01_raw/stocks_information_raw_data.csv'
    if Path(p).exists():
        print('Data already downloaded')
        return pd.read_csv('data/01_raw/stocks_information_raw_data.csv')
    
    print('Downloading stocks information from Yahoo')
    df = download_sp500() # The first table is the list
    return df

def download_sp500() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # Without a proper User-Agent Wikipedia sometimes returns a stripped page
    html = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
    ).text

    tables = pd.read_html(html)

    # Find the table that contains the 'Symbol' column
    for table in tables:
        if "Symbol" in table.columns:
            df = table
            break
    else:
        raise ValueError("Could not find S&P 500 table in Wikipedia page.")

    # Clean column names (optional)
    df.columns = [c.replace("\n", " ").strip() for c in df.columns]

    return df

def download_stocks_history(stocksInformationDf: pd.DataFrame) -> pd.DataFrame:
    p = os.getcwd() + '/data/01_raw/stocks_history_raw_data.csv'
    if Path(p).exists():
        print('Data already downloaded')
        return pd.read_csv('data/01_raw/stocks_history_raw_data.csv')
    
    print('Downloading stocks history from Yahoo')
    tickers = stocksInformationDf['Symbol'].tolist()

    data = pd.DataFrame()
    for ticker in tqdm(tickers):
        t = yf.Ticker(ticker)
        hist = t.history(start='2000-01-01', end='2025-11-01')
        
        if not hist.empty:
            # extract Date column from the index
            hist['Date'] = hist.index
            hist = hist[['Date'] + [col for col in hist.columns if col != 'Date']]

            hist['Ticker'] = ticker

            data = pd.concat([data, hist], ignore_index=True)
        
        time.sleep(1)
    
    return data