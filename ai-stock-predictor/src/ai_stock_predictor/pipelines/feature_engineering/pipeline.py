from kedro.pipeline import Pipeline, node, pipeline
from .nodes import add_lag_features, add_technical_indicators, add_date_features, add_company_features, \
    add_corporate_action_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=add_date_features,
            inputs="stocks_data_merged",
            outputs="stocks_with_date",
            name="date_features_node"
        ),
        node(
            func=add_company_features,
            inputs="stocks_with_date",
            outputs="stocks_with_company",
            name="company_features_node"
        ),
        node(
            func=add_lag_features,
            inputs=["stocks_with_company", "params:features.lag_days"],
            outputs="stocks_with_lags",
            name="lag_features_node"
        ),
        node(
            func=add_corporate_action_features,
            inputs="stocks_with_lags",
            outputs="stocks_with_corporate",
            name="corporate_features_node"
        ),
        node(
            func=add_technical_indicators,
            inputs="stocks_with_corporate",
            outputs="stocks_final",
            name="technical_indicators_node"
        ),
    ])