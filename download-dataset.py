import pandas as pd
import yfinance as yf
from pyfinance import TSeries

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
tables = pd.read_html(url)
df = tables[0]  # The first table is the list

# Get the Symbol column (list of tickers)
tickers = df['Symbol'].tolist()

data = yf.download(tickers)

# ts_data = {ticker: TSeries(data[ticker].dropna()) for ticker in tickers}

# df = pd.concat(data, axis=1)

df.to_csv('stocks.csv')