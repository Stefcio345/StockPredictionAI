import pandas as pd

def merge_both_datasets(stocksInformationDf: pd.DataFrame, stocksHistoryDf: pd.DataFrame) -> pd.DataFrame:
    merged =  pd.merge(stocksHistoryDf, stocksInformationDf, left_on='Ticker', right_on='Symbol')
    merged = merged.drop(columns='Ticker') # drop one of the Ticker/Symbol columns
    print(merged.head(3))
    return merged