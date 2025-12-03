import os
from datetime import date, timedelta
from functools import lru_cache
from typing import List, Dict

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from Azure_utils import download_model

# Kedro feature engineering imports (your existing code)
from ai_stock_predictor.src.ai_stock_predictor.pipelines.feature_engineering.nodes import (
    add_technical_indicators,
    add_lag_features,
    add_date_features,
    add_company_features,
    add_corporate_action_features,
)
from ai_stock_predictor.src.ai_stock_predictor.pipelines.data_merge.nodes import (
    merge_both_datasets,
)
from ai_stock_predictor.src.ai_stock_predictor.pipelines.data_cleaning.nodes import (
    convert_date_columns,
    clean_founded_column,
    remove_unused_columns,
)

# ==========================================================
# Pydantic models
# ==========================================================

class PredictRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker, e.g. AAPL")
    start_date: date = Field(..., description="Start date for prediction (YYYY-MM-DD)")
    days: int = Field(..., gt=0, le=90, description="Number of trading days to predict")

class OHLCPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float

class PredictResponse(BaseModel):
    ticker: str
    predictions: List[OHLCPoint]
    real: List[OHLCPoint]


# ==========================================================
# FastAPI app & CORS
# ==========================================================

app = FastAPI(
    title="AI Stock Predictor Backend",
    description="FastAPI backend serving Kedro-trained S&P 500 stock predictions.",
    version="1.0.0",
)

# Allow frontend on any origin (lock this down later if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# Data download / preprocessing (adapted from Streamlit code)
# ==========================================================

@lru_cache(maxsize=1)
def download_sp500() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # Without a proper User-Agent Wikipedia sometimes returns a stripped page
    html = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
    ).text

    tables = pd.read_html(html)

    # Find the table that contains the 'Symbol' column
    for table in tables:
        if "Symbol" in table.columns:
            df = table
            break
    else:
        raise ValueError("Could not find S&P 500 table in Wikipedia page.")

    # Clean column names (optional)
    df.columns = [c.replace("\n", " ").strip() for c in df.columns]

    return df


@lru_cache(maxsize=128)
def download_single_stock_history(ticker: str) -> pd.DataFrame:
    """Historical OHLC data for a single ticker using yfinance."""
    t = yf.Ticker(ticker)
    # You can adjust end date if you want strictly up-to-some-date behaviour
    hist = t.history(start="2000-01-01")
    hist = hist.reset_index()
    hist["Ticker"] = ticker
    return hist


def process_after_download(info_df: pd.DataFrame, history_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Processes and merges info and history DataFrames for a given ticker."""
    info_df, history_df = remove_unused_columns(info_df, history_df, ticker)
    history_df = convert_date_columns(history_df)
    info_df = clean_founded_column(info_df)
    merged_df = merge_both_datasets(info_df, history_df)
    return merged_df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline, same as in your Streamlit app."""
    df = add_technical_indicators(df)
    df = add_lag_features(df, lag_days=[1, 2, 3, 5, 10])
    df = add_date_features(df)
    df = add_company_features(df)
    df = add_corporate_action_features(df)
    return df


# ==========================================================
# Model loading (same logic as Streamlit, but without st)
# ==========================================================

def load_predictors() -> Dict[str, object]:
    """
    Load multi-target predictors.
    Tries Azure_downloaded file first, then local Kedro models as fallback.
    """
    model_path_dl = "downloaded_model.pkl"
    model_path_local = "./ai_stock_predictor/data/05_models/trained_multi_target_models.pkl"

    # Download from Azure if not present
    if not os.path.exists(model_path_dl):
        try:
            print("[model] Downloading model from Azure...")
            download_model()
        except Exception as e:
            print("[model] Failed to download from Azure:", e)

    # Try downloaded model
    if os.path.exists(model_path_dl):
        print("[model] Loading model from:", model_path_dl)
        predictors = joblib.load(model_path_dl)
        return predictors

    # Fallback: local kedro path
    if os.path.exists(model_path_local):
        print("[model] Loading model from:", model_path_local)
        predictors = joblib.load(model_path_local)
        return predictors

    raise RuntimeError("No model file found (downloaded_model.pkl or Kedro model).")


PREDICTORS: Dict[str, object] = {}


@app.on_event("startup")
def startup_event():
    """
    Initialize model once at startup.
    """
    global PREDICTORS
    PREDICTORS = load_predictors()
    print(f"[startup] Loaded predictors for targets: {list(PREDICTORS.keys())}")


# ==========================================================
# Core prediction logic (adapted from your Streamlit code)
# ==========================================================

def run_model_predict(predictors: Dict[str, object], input_df: pd.DataFrame) -> Dict[str, float]:
    """
    Run all target models on a (batched) input_df.
    Mirrors your `predict` function from Streamlit.
    """
    results = {}
    # Remove any *_target columns so multi-target regressors don't choke
    cleaned = input_df.drop(
        columns=[col for col in input_df.columns if col.endswith("_target")],
        errors="ignore",
    )

    for label, predictor in predictors.items():
        pred = predictor.predict(cleaned)  # usually returns numpy array or pandas
        # label like "Open_target" -> "Open"
        out_label = label.replace("_target", "")
        # use first row (we batch 2 rows just to satisfy model)
        results[out_label] = float(np.round(pred[0], 2))
    return results


def recursive_predict(
    ticker: str,
    start_date: date,
    days: int,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Full recursive prediction, returning:
      prediction_df (future OHLC predictions)
      real_df       (real OHLC in the same date range, for comparison)
    Logic mirrors your Streamlit implementation.
    """
    # 1. Download data
    info_df = download_sp500()
    if ticker not in info_df["Symbol"].values:
        raise HTTPException(status_code=400, detail="Ticker not found in S&P 500 list.")

    history_df = download_single_stock_history(ticker)
    if history_df.empty:
        raise HTTPException(status_code=400, detail="No historical data available for this ticker.")

    # 2. Merge & preprocess
    merged = process_after_download(info_df, history_df, ticker)

    # Make sure Date is date-type
    if not np.issubdtype(merged["Date"].dtype, np.datetime64):
        merged["Date"] = pd.to_datetime(merged["Date"])

    merged["Date"] = merged["Date"].dt.date

    # If exact start_date is not present, snap back to previous available trading day
    if merged[merged["Date"] == start_date].empty:
        possible = merged[merged["Date"] < start_date]
        if possible.empty:
            raise HTTPException(
                status_code=400,
                detail="Requested start_date is before any available data for this ticker.",
            )
        start_date = possible["Date"].max()

    # Keep history up to start_date
    history_until_now = merged[merged["Date"] <= start_date].copy()

    predictions = []
    current_date = start_date

    for _ in range(days):
        # 1. Full preprocessing on rolling dataset
        processed = preprocess(history_until_now)

        # 2. Take last row for prediction
        last_date = processed["Date"].max()
        current_row = processed[processed["Date"] == last_date].copy()
        if current_row.empty:
            raise HTTPException(
                status_code=500,
                detail=f"No data available to preprocess for {current_date}",
            )

        # 3. Create "batched" input (your workaround for the multi-target model)
        empty_row = current_row.iloc[0:1].copy()
        for col in empty_row.columns:
            if pd.api.types.is_integer_dtype(empty_row[col]):
                empty_row[col] = empty_row[col].astype("float")
            elif pd.api.types.is_datetime64_any_dtype(empty_row[col]):
                empty_row[col] = pd.NaT
            else:
                empty_row[col] = empty_row[col].astype("object")
        empty_row.loc[:] = np.nan
        batched = pd.concat([current_row, empty_row], ignore_index=True)

        # 4. Predict
        pred_values = run_model_predict(PREDICTORS, batched)

        # 5. Create new pseudo-future row (using predicted OHLC)
        new_row = current_row.copy()
        for k, v in pred_values.items():
            if k in new_row.columns:
                new_row[k] = v
        # we set the new row's Date to *next* trading day
        next_date = current_date
        while True:
            next_date = next_date + timedelta(days=1)
            if next_date.weekday() < 5:  # Mon-Fri only
                break
        new_row["Date"] = next_date

        # 6. Save prediction for current_date (like in your original loop)
        prediction_entry = {
            "Date": current_date,
            "Open": pred_values.get("Open", np.nan),
            "High": pred_values.get("High", np.nan),
            "Low": pred_values.get("Low", np.nan),
            "Close": pred_values.get("Close", np.nan),
        }
        predictions.append(prediction_entry)

        # 7. Append new_row to rolling history and move on
        history_until_now = pd.concat([history_until_now, new_row], ignore_index=True)
        current_date = next_date

    prediction_df = pd.DataFrame(predictions)

    # Extract real data for comparison (same range as predictions)
    start_range = prediction_df["Date"].min()
    end_range = prediction_df["Date"].max()
    real_df = merged[(merged["Date"] >= start_range) & (merged["Date"] <= end_range)].copy()

    return prediction_df, real_df


# ==========================================================
# API endpoints
# ==========================================================

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": bool(PREDICTORS)}


@app.post("/api/predict", response_model=PredictResponse)
def api_predict(req: PredictRequest):
    ticker = req.ticker.upper()

    if not PREDICTORS:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    prediction_df, real_df = recursive_predict(
        ticker=ticker,
        start_date=req.start_date,
        days=req.days,
    )

    # Normalize to Python types / strings for JSON
    def df_to_ohlc(df: pd.DataFrame) -> List[Dict]:
        records = []
        for _, row in df.iterrows():
            records.append(
                {
                    "date": row["Date"].isoformat()
                    if isinstance(row["Date"], date)
                    else str(row["Date"]),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                }
            )
        return records

    predictions = df_to_ohlc(prediction_df)
    real = df_to_ohlc(real_df) if not real_df.empty else []

    return PredictResponse(
        ticker=ticker,
        predictions=predictions,
        real=real,
    )
