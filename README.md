# StockPredictionAI

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

AI-powered S&P 500 price forecasting that combines a Kedro feature-engineering pipeline, FastAPI inference service, and a sleek Chart.js frontend. The backend downloads company metadata and price history, engineers predictive features, and serves recursive multi-day OHLC forecasts through a single endpoint consumed by the interactive dashboard in `frontend/`.

## Features
- **Production-ready FastAPI backend:** Serves `/health` and `/api/predict` endpoints with CORS enabled for the bundled frontend.
- **Kedro feature engineering:** Reuses the project's pipelines for technical indicators, lagged signals, calendar features, and corporate action context before inference.
- **Multi-target forecasting:** Loads pretrained models (downloaded from Azure Blob Storage or local Kedro artifacts) to predict Open/High/Low/Close simultaneously.
- **Automatic data sourcing:** Scrapes the latest S&P 500 constituents from Wikipedia and fetches historical OHLC data via Yahoo Finance.
- **Polished frontend experience:** `frontend/index.html` provides a neon-styled dashboard with ticker search, date pickers, and Chart.js overlays comparing forecasts against real prices.

## Project layout
- `StockPredictionApp.py` – FastAPI application with request/response schemas, model loading, and recursive prediction loop.
- `Azure_utils.py` – Helper for downloading the trained multi-target model from Azure Blob Storage to `downloaded_model.pkl`.
- `frontend/` – Static HTML/JS dashboard (defaults to `http://127.0.0.1:8000/api/predict` for API calls).
- `ai_stock_predictor/` – Kedro project containing the feature-engineering pipelines and training artifacts.
- `requirements.txt` – Full Python dependency lock for backend, frontend tooling, and Kedro notebooks.

## Getting started
1. **Prerequisites:** Python 3.10+ is recommended.
2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Provide the model artifact:**
   - Set your Azure Blob Storage connection string in `Azure_utils.py` and run `download_model()` once, **or**
   - Place a pretrained `downloaded_model.pkl` (or the Kedro-generated `ai_stock_predictor/data/05_models/trained_multi_target_models.pkl`) in the repository root.

## Running the backend
Start the FastAPI server with Uvicorn:
```bash
uvicorn StockPredictionApp:app --reload --host 0.0.0.0 --port 8000
```
- `GET /health` returns a quick readiness check.
- `POST /api/predict` accepts ticker, start date, and number of trading days to forecast.

### Example request
```bash
curl -X POST "http://127.0.0.1:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "ticker": "AAPL",
        "start_date": "2024-01-02",
        "days": 10
      }'
```

## Running the frontend
Open the dashboard served from the `frontend/` directory:
```bash
python -m http.server 3000 --directory frontend
```
Then visit `http://127.0.0.1:3000` in your browser. If your backend runs on a different host or port, update the fetch URL inside `frontend/index.html` (search for `/api/predict`).

## Working with the Kedro project
Inside `ai_stock_predictor/` you can run the original pipelines and notebooks:
```bash
cd ai_stock_predictor
kedro run       # execute the configured pipelines
kedro test      # run project tests
```
The FastAPI app imports pipeline nodes directly, so changes to feature engineering will also flow into API predictions.

## Troubleshooting
- **Model not loaded:** Ensure `downloaded_model.pkl` exists or that Azure credentials are set correctly before starting the server.
- **Ticker validation:** Only tickers present in the latest S&P 500 list from Wikipedia are accepted.
- **Date alignment:** The service automatically snaps the requested `start_date` back to the latest available trading day when necessary.

## License
This project is licensed under the terms of the [LICENSE](LICENSE) file.
