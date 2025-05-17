import pandas as pd
import yfinance as yf
from pyfinance import TSeries
from tqdm import tqdm
import time

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
tables = pd.read_html(url)
df = tables[0]  # The first table is the list
df.to_csv('stockpredictionai/data/01_raw/stocks-information.csv', index=False)

# Get the Symbol column (list of tickers)
tickers = df['Symbol'].tolist()

data = pd.DataFrame()
for ticker in tqdm(tickers):
    t = yf.Ticker(ticker)
    hist = t.history(start='2000-01-01', end='2025-04-25')
    hist['Ticker'] = ticker
    data = pd.concat([data, hist], ignore_index=True)
    time.sleep(1)

# ts_data = {ticker: TSeries(data[ticker].dropna()) for ticker in tickers}

# df = pd.concat(data, axis=1)

print(data.head())

data.to_csv('stockpredictionai/data/01_raw/stocks-history.csv', index=False)