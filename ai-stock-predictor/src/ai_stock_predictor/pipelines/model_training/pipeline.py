from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data, train_multi_target_models, evaluate_multi_target_models

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs=["stocks_final", "params:target_columns", "params:ticker"],
            outputs=["train_data", "test_data"],
            name="split_data_node",
        ),
        node(
            func=train_multi_target_models,
            inputs=["train_data", "params:target_columns"],
            outputs="trained_multi_target_models",
            name="train_multi_target_models_node",
        ),
        node(
            func=evaluate_multi_target_models,
            inputs=["trained_multi_target_models", "test_data", "params:target_columns"],
            outputs="multi_target_metrics",
            name="evaluate_multi_target_models_node",
        ),
    ])