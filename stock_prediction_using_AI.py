import streamlit as st
import os
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from textblob import TextBlob
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv("api_keys.env")
FMP_API_KEY = os.getenv("FMP_API_KEY")

# Initialize model globally
model = None  

# Fetch stock data from API
def fetch_stock_data(symbol, start_date, end_date):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&to={end_date}&apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200 and "historical" in response.json():
        df = pd.DataFrame(response.json()["historical"])
        df = df[["date", "close"]]
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df
    return None

# Prepare Data for LSTM Model
def prepare_lstm_data(stock_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data["close"].values.reshape(-1, 1))

    X, y = [], []
    for i in range(60, len(scaled_data)):  # Use last 60 days to predict next day
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM input
    return X, y, scaler

# Train and Save LSTM Model
def train_lstm_model(X_train, y_train):
    global model  # Ensure model is accessible globally
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)  # Train model
    model.save("stock_forecast_model.h5")  # Save trained model
    return model

# Load pre-trained LSTM model (if available)
def load_or_train_model(X_train, y_train):
    global model  
    try:
        model = load_model("stock_forecast_model.h5")
        model.compile(optimizer="adam", loss="mean_squared_error")  # Ensure it's compiled
    except:
        st.warning("No pre-trained model found. Training a new model...")
        model = train_lstm_model(X_train, y_train)

# Predict stock prices using LSTM
def predict_future_prices(stock_history, scaler):
    if model is None or stock_history is None:
        return None

    last_60_days = stock_history["close"].values[-60:].reshape(-1, 1)
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = np.reshape(last_60_days_scaled, (1, 60, 1))

    predicted_price_scaled = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)  # Convert back to actual price
    return predicted_price[0][0]

# Sentiment analysis
def analyze_sentiment(news_text):
    sentiment_score = TextBlob(news_text).sentiment.polarity
    if sentiment_score > 0:
        return "📈 Positive"
    elif sentiment_score < 0:
        return "📉 Negative"
    else:
        return "⚖️ Neutral"

# Show Stock Prediction Page
def show_stock_predictionfmp():
    st.title("Stock Market Prediction Using AI & ML 🧠💹")
    st.sidebar.header("User Input Parameters")
    
    top_companies = {"Apple Inc. (AAPL)": "AAPL", "Microsoft Corporation (MSFT)": "MSFT", "Amazon.com, Inc. (AMZN)": "AMZN"}
    selected_company = st.sidebar.selectbox("Select a Company", list(top_companies.keys()))
    ticker = top_companies[selected_company]
    start_date = st.sidebar.date_input("Select Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("Select End Date", pd.to_datetime("2025-01-01"))
    
    st.write(f"Fetching data for {ticker}...")
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    if stock_data is None or len(stock_data) < 60:
        st.error("Stock data not available or insufficient data. Please check the ticker or try again later.")
        return
    
    st.write(stock_data.head())

    # Prepare LSTM data
    X, y, scaler = prepare_lstm_data(stock_data)
    X_train, y_train = X[:int(0.8 * len(X))], y[:int(0.8 * len(y))]  # Split data for training

    # Load model or train a new one if not found
    load_or_train_model(X_train, y_train)
    
    # Predict Future Prices
    future_price = predict_future_prices(stock_data, scaler)
    if future_price:
        st.metric(label=f"📈 Predicted Price for {ticker} (Next Day)", value=f"${future_price:.2f}")

    # Sentiment Analysis (Example)
    example_news = "Stock market surges as investors remain optimistic about economic recovery."
    sentiment = analyze_sentiment(example_news)
    st.write(f"📰 Market Sentiment: {sentiment}")

    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["close"], mode="lines", name="Closing Price"))
    fig.update_layout(title=f"Stock Price Trend for {ticker}", xaxis_title="Date", yaxis_title="Price ($)")
    st.plotly_chart(fig)

# Footer
def add_footer():
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            color: #21130d;
        }
        </style>
        <div class="footer">
            © 2025 StockMarketIQ. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Run the application
if __name__ == "__main__":
    show_stock_predictionfmp()
    add_footer()
