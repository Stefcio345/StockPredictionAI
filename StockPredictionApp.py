import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import plotly.graph_objects as go
from datetime import timedelta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['Typical price'] = (df['High'] + df['Low'] + df['Close']) / 3

    # Added grouping by company
    df["MA_10"] = df.groupby("Symbol")["Close"].transform(lambda x: x.rolling(10).mean())
    df["MA_10"] = df.groupby("Symbol")["Close"].transform(lambda x: x.rolling(50).mean())
    df["RSI"] = compute_rsi_grouped(df)

    return df

def add_lag_features(df: pd.DataFrame, lag_days: list[int]) -> pd.DataFrame:
    df = df.copy()
    for lag in lag_days:
        df[f"lag_{lag}"] = df.groupby("Symbol")["Close"].shift(lag)

    # Example: lag RSI and MA_10 if they exist
    if 'RSI' in df.columns:
        for lag in [1, 2]:
            df[f"RSI_lag_{lag}"] = df.groupby("Symbol")["RSI"].shift(lag)
    if 'MA_10' in df.columns:
        for lag in [1, 2]:
            df[f"MA_10_lag_{lag}"] = df.groupby("Symbol")["MA_10"].shift(lag)

    return df

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['Date'] = pd.to_datetime(df['Date'])

    df['day_of_week'] = df['Date'].dt.dayofweek       # 0=Monday
    df['day_of_month'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)

    return df

def add_company_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure datetime types
    df['Date'] = pd.to_datetime(df['Date'])
    df['Founded'] = pd.to_datetime(df['Founded'], errors='coerce')
    df['Date added'] = pd.to_datetime(df['Date added'], errors='coerce')

    # Company age at each row
    df['company_age'] = df['Date'].dt.year - df['Founded'].dt.year

    # Years since added to index (e.g. S&P500)
    df['years_since_added'] = df['Date'].dt.year - df['Date added'].dt.year

    # Handle cases where founding date or date added is missing
    df['company_age'] = df['company_age'].fillna(-1)
    df['years_since_added'] = df['years_since_added'].fillna(-1)

    return df

def add_corporate_action_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['dividend_lag_1'] = df.groupby("Symbol")["Dividends"].shift(1)
    df['split_lag_1'] = df.groupby("Symbol")["Stock Splits"].shift(1)
    return df

def compute_rsi_grouped(df: pd.DataFrame, window: int = 14) -> pd.Series:
    def compute_rsi(series: pd.Series) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    return df.groupby("Symbol")["Close"].transform(compute_rsi)

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

def merge_both_datasets(info_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(history_df, info_df, left_on="Ticker", right_on="Symbol")
    return merged.drop(columns="Ticker")

def clean_founded_column(df: pd.DataFrame):
    df["Founded"] = df["Founded"].astype(str).str.extract(r"(\d{4})")
    return df

# === Full preprocessing #TODO: Copy this from kedro
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = add_technical_indicators(df)
    df = add_lag_features(df, lag_days=[1, 2, 3, 5, 10])
    df = add_date_features(df)
    df = add_company_features(df)
    df = add_corporate_action_features(df)
    return df

# === Load trained models
predictors = joblib.load("predictors.pkl")

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

if st.button("Predict"):
    with st.spinner("Downloading and processing data..."):
        info_df = download_stocks_information()
        history_df = download_single_stock_history(ticker_input)

        if history_df.empty or ticker_input not in info_df["Symbol"].values:
            st.error("Invalid ticker or no data available.")
        else:
            info_df = info_df[info_df["Symbol"] == ticker_input]
            info_df = clean_founded_column(info_df)
            merged = merge_both_datasets(info_df, history_df)
            merged = preprocess(merged)

            merged["Date"] = pd.to_datetime(merged["Date"]).dt.date

            row = merged[merged["Date"] == date_input]

            if row.empty:
                st.warning("No data available for that ticker on the selected date.")
            else:
                st.success("Starting Prediction")

                # Start recursive prediction
                predictions = []
                start_date = date_input
                end_date = max(merged["Date"])

                # Keep original processed data until start_date
                history_until_now = merged[merged["Date"] <= start_date].copy()

                print(start_date, end_date)

                while start_date <= end_date:

                    # 1. Run full preprocessing on the rolling dataset
                    processed = preprocess(history_until_now)
                    processed["Date"] = pd.to_datetime(processed["Date"]).dt.date

                    # 2. Get last row for prediction
                    current_row = processed[processed["Date"] == start_date]
                    if current_row.empty:
                        st.warning(f"No data available to preprocess on {start_date}")
                        break

                    # 3. Predict
                    prediction = predict(predictors, current_row)

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

                # Predicted close as line
                fig.add_trace(go.Scatter(
                    x=prediction_df["Date"],
                    y=prediction_df["Close"],
                    mode='lines+markers',
                    name="Predicted Close",
                    line=dict(color='lime', width=2, dash='dash')
                ))

                fig.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)

                st.write(prediction_df)