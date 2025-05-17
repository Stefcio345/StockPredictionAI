from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    prepare_model_input, split_data,
    train_model, evaluate_model, make_predictions
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_model_input,
            inputs="stock_features",
            outputs="model_input",
            name="prepare_input_node"
        ),
        node(
            func=split_data,
            inputs=["model_input", "params:model_params.test_size", "params:model_params.random_state"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_node"
        ),
        node(
            func=train_model,
            inputs=["X_train", "y_train", "params:model_params.xgboost"],
            outputs="trained_model",
            name="train_model_node"
        ),
        node(
            func=evaluate_model,
            inputs=["trained_model", "X_test", "y_test"],
            outputs="model_metrics",
            name="evaluate_model_node"
        ),
        node(
            func=make_predictions,
            inputs=["trained_model", "X_test"],
            outputs="model_predictions",
            name="make_predictions_node"
        )
    ])