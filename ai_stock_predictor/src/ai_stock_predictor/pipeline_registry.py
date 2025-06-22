from kedro.pipeline import Pipeline
from ai_stock_predictor.pipelines.download_data import pipeline as download_data_pipeline
from ai_stock_predictor.pipelines.feature_engineering import pipeline as feature_engineering_pipeline
from ai_stock_predictor.pipelines.data_cleaning import pipeline as data_cleaning_pipeline
from ai_stock_predictor.pipelines.data_merge import pipeline as data_merge_pipeline
from ai_stock_predictor.pipelines.model_training import pipeline as model_training_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "download_data": download_data_pipeline.create_pipeline(),
        "data_cleaning": data_cleaning_pipeline.create_pipeline(),
        "data_merge": data_merge_pipeline.create_pipeline(),
        "feature_engineering": feature_engineering_pipeline.create_pipeline(),
        "model_training": model_training_pipeline.create_pipeline(),

        "__default__": (
            download_data_pipeline.create_pipeline()
            + data_cleaning_pipeline.create_pipeline()
            + data_merge_pipeline.create_pipeline()
            + feature_engineering_pipeline.create_pipeline()
            + model_training_pipeline.create_pipeline()
        )
    }