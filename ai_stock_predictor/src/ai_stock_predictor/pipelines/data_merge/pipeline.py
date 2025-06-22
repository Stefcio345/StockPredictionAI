from kedro.pipeline import Pipeline, node, pipeline
from .nodes import merge_both_datasets

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=merge_both_datasets,
            inputs=['stocks_information_cleaned', 'stocks_history_cleaned'],
            outputs='stocks_data_merged',
            name='merge_both_datasets'
        )
    ])