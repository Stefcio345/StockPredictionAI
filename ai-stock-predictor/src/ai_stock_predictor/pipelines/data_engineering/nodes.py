import yfinance as yf
import pandas as pd
from tqdm import tqdm
import time
from pathlib import Path
import os

def download_stocks_information() -> pd.DataFrame:
    p = os.getcwd() + '/data/01_raw/stocks_information_raw_data.csv'
    if Path(p).exists():
        print('Data already downloaded')
        return pd.read_csv('data/01_raw/stocks_information_raw_data.csv')
    
    print('Downloading stocks information from Yahoo')
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]  # The first table is the list
    return df

def download_stocks_history(stocksInformationDf: pd.DataFrame) -> pd.DataFrame:
    p = os.getcwd() + '/data/01_raw/stocks_history_raw_data.csv'
    if Path(p).exists():
        print('Data already downloaded')
        print(data.head(1))
        return pd.read_csv('data/01_raw/stocks_history_raw_data.csv')
    
    print('Downloading stocks history from Yahoo')
    tickers = stocksInformationDf['Symbol'].tolist()

    data = pd.DataFrame()
    for ticker in tqdm(tickers):
        t = yf.Ticker(ticker)
        hist = t.history(start='2000-01-01', end='2025-06-07')
        
        # extract Date column from the index
        hist['Date'] = hist.index
        hist = hist[['Date'] + [col for col in hist.columns if col != 'Date']]

        hist['Ticker'] = ticker

        data = pd.concat([data, hist], ignore_index=True)
        time.sleep(1)
    
    return data