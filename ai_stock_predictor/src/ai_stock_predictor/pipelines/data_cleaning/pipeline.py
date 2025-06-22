from kedro.pipeline import Pipeline, node, pipeline
from .nodes import remove_unused_columns, convert_date_columns, clean_founded_column

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=remove_unused_columns,
            inputs=['stocks_information_raw_data', 'stocks_history_raw_data', "params:ticker"],
            outputs=['stocks_information_removed_unused', 'stocks_history_removed_unused'],
            name='remove_unused_columns_node'
        ),
        node(
            func=convert_date_columns,
            inputs=['stocks_history_removed_unused'],
            outputs='stocks_history_cleaned',
            name='convert_date_columns'
        ),
        node(
            func=clean_founded_column,
            inputs=['stocks_information_removed_unused'],
            outputs='stocks_information_cleaned',
            name='clean_founded_column'
        )
    ])