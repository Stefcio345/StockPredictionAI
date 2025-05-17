from kedro.pipeline import Pipeline, node, pipeline
from .nodes import download_stock_data, clean_stock_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=download_stock_data,
            inputs=["params:data_params.ticker", "params:data_params.start_date", "params:data_params.end_date"],
            outputs="stock_raw_data",
            name="download_data_node"
        ),
        node(
            func=clean_stock_data,
            inputs=["stock_raw_data", "params:preprocessing.fillna_method"],
            outputs="stock_cleaned_data",
            name="clean_data_node"
        ),
    ])