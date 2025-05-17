from kedro.pipeline import Pipeline
from ai_stock_predictor.pipelines.data_engineering import pipeline as de_pipeline
from ai_stock_predictor.pipelines.feature_engineering import pipeline as fe_pipeline
from ai_stock_predictor.pipelines.data_modeling import pipeline as m_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "de": de_pipeline.create_pipeline(),
        "fe": fe_pipeline.create_pipeline(),
        "model": m_pipeline.create_pipeline(),
        "__default__": de_pipeline.create_pipeline() + fe_pipeline.create_pipeline() + m_pipeline.create_pipeline()
    }