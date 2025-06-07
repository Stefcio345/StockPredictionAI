from kedro.pipeline import Pipeline
from ai_stock_predictor.pipelines.download_data import pipeline as download_data_pipeline
from ai_stock_predictor.pipelines.feature_engineering import pipeline as fe_pipeline
from ai_stock_predictor.pipelines.data_modeling import pipeline as m_pipeline
from ai_stock_predictor.pipelines.data_cleaning import pipeline as data_cleaning_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "download_data": download_data_pipeline.create_pipeline(),
        "data_cleaning": data_cleaning_pipeline.create_pipeline(),
        # "fe": fe_pipeline.create_pipeline(),
        # "model": m_pipeline.create_pipeline(),
        "__default__": download_data_pipeline.create_pipeline() + data_cleaning_pipeline.create_pipeline() #+ fe_pipeline.create_pipeline() + m_pipeline.create_pipeline()
    }