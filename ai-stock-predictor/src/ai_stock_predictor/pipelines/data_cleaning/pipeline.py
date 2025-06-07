from kedro.pipeline import Pipeline, node, pipeline
from .nodes import remove_unused_columns

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=remove_unused_columns,
            inputs=['stocks_information_raw_data', 'stocks_history_raw_data'],
            outputs=['stocks_information_cleaned', 'stocks_history_cleaned'],
            name='remove_unused_columns_node'
        )
    ])