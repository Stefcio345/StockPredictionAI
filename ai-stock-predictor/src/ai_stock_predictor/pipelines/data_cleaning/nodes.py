import pandas as pd
from typing import Tuple

def remove_unused_columns(stocksInformationDf: pd.DataFrame, stocksHistoryDf: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stocksInformationDf = stocksInformationDf.drop(columns=['Security', 'CIK'])

    stocksHistoryDf = stocksHistoryDf.drop(columns=['Adj Close'])

    # Drop headquarters location due to a high number of unique values
    stocksInformationDf = stocksInformationDf.drop(columns='Headquarters Location')

    return stocksInformationDf, stocksHistoryDf

def convert_date_columns(stocksHistoryDf: pd.DataFrame) -> pd.DataFrame:
    stocksHistoryDf['Date'] = pd.to_datetime(stocksHistoryDf['Date'], utc=True, errors='coerce').dt.date
    return stocksHistoryDf

def clean_founded_column(stocksInformationDf):
    stocksInformationDf['Founded'] = stocksInformationDf['Founded'].map(lambda x: x.split(' ')[0] if len(x.split(' ')) > 1 else x)
    stocksInformationDf['Founded'] = stocksInformationDf['Founded'].map(lambda x: x.split('/')[0] if len(x.split('/')) > 1 else x)
    
    return stocksInformationDf