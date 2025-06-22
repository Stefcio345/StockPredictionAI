import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import plotly.graph_objects as go
import numpy as np
from autogluon.multimodal import MultiModalPredictor

from ai_stock_predictor.src.ai_stock_predictor.pipelines.feature_engineering.nodes import add_technical_indicators
from ai_stock_predictor.src.ai_stock_predictor.pipelines.feature_engineering.nodes import add_lag_features
from ai_stock_predictor.src.ai_stock_predictor.pipelines.feature_engineering.nodes import add_date_features
from ai_stock_predictor.src.ai_stock_predictor.pipelines.feature_engineering.nodes import add_company_features
from ai_stock_predictor.src.ai_stock_predictor.pipelines.feature_engineering.nodes import add_corporate_action_features

from ai_stock_predictor.src.ai_stock_predictor.pipelines.data_merge.nodes import merge_both_datasets

from ai_stock_predictor.src.ai_stock_predictor.pipelines.data_cleaning.nodes import convert_date_columns
from ai_stock_predictor.src.ai_stock_predictor.pipelines.data_cleaning.nodes import clean_founded_column
from ai_stock_predictor.src.ai_stock_predictor.pipelines.data_cleaning.nodes import remove_unused_columns

from ai_stock_predictor.src.ai_stock_predictor.pipelines.model_training.nodes import split_data



from datetime import timedelta

@st.cache_data
def download_stocks_information():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    return tables[0]

@st.cache_data
def download_single_stock_history(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    hist = t.history(start="2000-01-01", end="2025-06-18")
    hist = hist.reset_index()
    hist["Ticker"] = ticker
    return hist

def process_after_download(info_df: pd.DataFrame, history_df: pd.DataFrame, ticker: str):
    """Processes and merges info and history DataFrames for a given ticker."""
    info_df, history_df = remove_unused_columns(info_df, history_df, ticker)
    history_df = convert_date_columns(history_df)
    info_df = clean_founded_column(info_df)
    merged_df = merge_both_datasets(info_df, history_df)
    return merged_df

# === Full preprocessing #TODO: Copy this from kedro
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = add_technical_indicators(df)
    df = add_lag_features(df, lag_days=[1, 2, 3, 5, 10])
    df = add_date_features(df)
    df = add_company_features(df)
    df = add_corporate_action_features(df)
    return df

# === Load trained models
predictors = joblib.load("./ai_stock_predictor/data/05_models/trained_multi_target_models.pkl")

# === Predict function
def predict(predictors: dict, input_df: pd.DataFrame) -> dict:
    results = {}
    for label, predictor in predictors.items():
        cleaned = input_df.drop(columns=[col for col in input_df.columns if col.endswith("_target")], errors="ignore")
        pred = predictor.predict(cleaned)

        results[label.replace("_target", "")] = round(pred.values[0], 2)
    return results

# === Streamlit UI
st.title("📈 S&P 500 Stock Price Predictor")

ticker_input = st.text_input("Enter stock ticker (e.g., AAPL, MSFT, GOOG):", value="AAPL").upper()
date_input = st.date_input("Select date to predict:", pd.to_datetime("2025-05-01"))
number_of_days_to_predict = st.number_input("input number of days to predcit:", 7)

if st.button("Predict"):
    with st.spinner("Downloading and processing data..."):
        info_df = download_stocks_information()
        history_df = download_single_stock_history(ticker_input)

        if history_df.empty or ticker_input not in info_df["Symbol"].values:
            st.error("Invalid ticker or no data available.")
        else:
            merged = process_after_download(info_df, history_df, ticker_input)
            merged = preprocess(merged)

            row = merged[merged["Date"].dt.date == date_input]

            if merged[merged["Date"].dt.date == date_input].empty:
                date_input = max(merged[merged["Date"].dt.date < date_input]["Date"].dt.date)
                st.warning("No data available on that date. Prediction will continue from the closest earlier available date: " + date_input)
            else:
                st.success("Starting Prediction")

                # Start recursive prediction
                predictions = []
                start_date = date_input
                merged["Date"] = merged["Date"].dt.date
                end_date = max(merged["Date"])

                # Keep original processed data until start_date
                history_until_now = merged[merged["Date"] <= start_date].copy()

                print(start_date, end_date)

                for _ in range(int(number_of_days_to_predict)):

                    # 1. Run full preprocessing on the rolling dataset
                    processed = preprocess(history_until_now)

                    # 2. Get last row for prediction
                    current_row = processed[processed["Date"].dt.normalize() == pd.to_datetime(start_date)].copy()

                    if current_row.empty:
                        st.warning(f"No data available to preprocess on {start_date}")
                        break

                    # Code specifically to add boilerplate data to prediciton as Multimodal does not work with single row of data
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

                    # 3. Predict
                    prediction = predict(predictors, batched)

                    # 4. Create fake future row from prediction
                    new_row = current_row.copy()

                    for k, v in prediction.items():
                        new_row[k] = v

                    new_row["Date"] = start_date + timedelta(days=1)

                    prediction["Date"] = start_date
                    predictions.append(prediction)

                    # 5. Append to rolling history
                    history_until_now = pd.concat([history_until_now, new_row], ignore_index=True)

                    # 6. Step forward
                    start_date += timedelta(days=1)


                prediction_df = pd.DataFrame(predictions)
                st.success("Recursive prediction complete.")

                # ======== PLOT CHARTS =======
                real_df = merged[(merged["Date"] >= prediction_df["Date"].min()) &
                                 (merged["Date"] <= prediction_df["Date"].max())]

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Predicted Candlestick - " + ticker_input)
                    fig_pred = go.Figure(data=[
                        go.Candlestick(
                            x=prediction_df["Date"],
                            open=prediction_df["Open"],
                            high=prediction_df["High"],
                            low=prediction_df["Low"],
                            close=prediction_df["Close"],
                            increasing_line_color='green',
                            decreasing_line_color='red',
                            name='Predicted'
                        )
                    ])
                    fig_pred.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Price")
                    st.plotly_chart(fig_pred, use_container_width=True)

                with col2:
                    st.subheader("Real Candlestick - " + ticker_input)
                    fig_real = go.Figure(data=[
                        go.Candlestick(
                            x=real_df["Date"],
                            open=real_df["Open"],
                            high=real_df["High"],
                            low=real_df["Low"],
                            close=real_df["Close"],
                            increasing_line_color='blue',
                            decreasing_line_color='orange',
                            name='Real'
                        )
                    ])
                    fig_real.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Price")
                    st.plotly_chart(fig_real, use_container_width=True)

                st.subheader("Real vs Predicted - " + ticker_input)

                fig = go.Figure()

                # Real candlestick
                fig.add_trace(go.Candlestick(
                    x=real_df["Date"],
                    open=real_df["Open"],
                    high=real_df["High"],
                    low=real_df["Low"],
                    close=real_df["Close"],
                    name="Real",
                    increasing_line_color='gray',
                    decreasing_line_color='dimgray'
                ))

                # Predicted typical price as line
                prediction_df['Typical price'] = (prediction_df['High'] + prediction_df['Low'] + prediction_df['Close']) / 3
                fig.add_trace(go.Scatter(
                    x=prediction_df["Date"],
                    y=prediction_df["Typical price"],
                    mode='lines+markers',
                    name="Predicted typical price",
                    line=dict(color='lime', width=2, dash='dash')
                ))

                fig.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)

                st.write(prediction_df)