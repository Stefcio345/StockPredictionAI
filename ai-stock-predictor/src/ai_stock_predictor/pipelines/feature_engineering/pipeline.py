from kedro.pipeline import Pipeline, node, pipeline
from .nodes import add_lag_features, add_technical_indicators

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=add_lag_features,
            inputs=["stocks_data_merged", "params:features.lag_days"],
            outputs="stocks_with_lags",
            name="lag_features_node"
        ),
        node(
            func=add_technical_indicators,
            inputs="stocks_with_lags",
            outputs="stocks_final",
            name="technical_indicators_node"
        )
    ])