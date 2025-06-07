import pandas as pd
from typing import Tuple

def remove_unused_columns(stocksInformationDf: pd.DataFrame, stocksHistoryDf: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stocksInformationDf = stocksInformationDf.drop(columns=['Security', 'CIK'])

    stocksHistoryDf = stocksHistoryDf.drop(columns=['Adj Close'])

    stocksInformationDf['Founded'] = stocksInformationDf['Founded'].map(lambda x: x.split(' ')[0] if len(x.split(' ')) > 1 else x)
    stocksInformationDf['Founded'] = stocksInformationDf['Founded'].map(lambda x: x.split('/')[0] if len(x.split('/')) > 1 else x)

    return stocksInformationDf, stocksHistoryDf