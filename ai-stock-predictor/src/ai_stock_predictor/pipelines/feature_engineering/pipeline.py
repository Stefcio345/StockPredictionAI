from kedro.pipeline import Pipeline, node, pipeline
from .nodes import add_lag_features, add_technical_indicators

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=add_lag_features,
            inputs=["stock_cleaned_data", "params:features.lag_days"],
            outputs="stock_with_lags",
            name="lag_features_node"
        ),
        node(
            func=add_technical_indicators,
            inputs="stock_with_lags",
            outputs="stock_features",
            name="technical_indicators_node"
        )
    ])