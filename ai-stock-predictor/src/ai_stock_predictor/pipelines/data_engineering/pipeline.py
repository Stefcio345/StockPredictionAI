from kedro.pipeline import Pipeline, node, pipeline
from .nodes import download_stocks_history, download_stocks_information

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=download_stocks_information,
            inputs=[],
            outputs="stocks_information_raw_data",
            name="download_stocks_info_node"
        ),
        node(
            func=download_stocks_history,
            inputs=["stocks_information_raw_data"],
            outputs="stocks_history_raw_data",
            name="download_stocks_history_node"
        )
    ])