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


class StockPredictor:
    def __init__(self):
        self.model = None  # Placeholder for the LSTM model
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fetch_stock_data(self, symbol, start_date, end_date):
        """Fetch historical stock data from API."""
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&to={end_date}&apikey={FMP_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200 and "historical" in response.json():
            df = pd.DataFrame(response.json()["historical"])
            df = df[["date", "close"]]
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            return df
        return None

    def prepare_lstm_data(self, stock_data):
        """Prepare stock data for LSTM model training."""
        scaled_data = self.scaler.fit_transform(stock_data["close"].values.reshape(-1, 1))

        X, y = [], []
        for i in range(60, len(scaled_data)):  # Use last 60 days to predict next day
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM input
        return X, y

    def train_lstm_model(self, X_train, y_train):
        """Train and save an LSTM model."""
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        self.model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
        self.model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        self.model.save("stock_forecast_model.h5")  # Save trained model

    def load_or_train_model(self, X_train, y_train):
        """Load pre-trained LSTM model or train a new one if not found."""
        try:
            self.model = load_model("stock_forecast_model.h5")
            self.model.compile(optimizer="adam", loss="mean_squared_error")
        except:
            st.warning("No pre-trained model found. Training a new model...")
            self.train_lstm_model(X_train, y_train)

    def predict_future_price(self, stock_history):
        """Predict the next day's stock price using LSTM."""
        if self.model is None or stock_history is None or len(stock_history) < 60:
            return None

        last_60_days = stock_history["close"].values[-60:].reshape(-1, 1)
        last_60_days_scaled = self.scaler.transform(last_60_days)
        X_test = np.reshape(last_60_days_scaled, (1, 60, 1))

        predicted_price_scaled = self.model.predict(X_test)
        predicted_price = self.scaler.inverse_transform(predicted_price_scaled)  # Convert back to actual price
        return predicted_price[0][0]

    def analyze_sentiment(self, news_text):
        """Perform sentiment analysis on financial news."""
        sentiment_score = TextBlob(news_text).sentiment.polarity
        if sentiment_score > 0:
            return "📈 Positive"
        elif sentiment_score < 0:
            return "📉 Negative"
        else:
            return "⚖️ Neutral"

    def show_stock_prediction_page(self):
        """Render Streamlit UI for stock prediction."""
        st.title("Stock Market Prediction Using AI & ML 🧠💹")
        st.sidebar.header("User Input Parameters")

        top_companies = {
            "Apple Inc. (AAPL)": "AAPL",
            "Microsoft Corporation (MSFT)": "MSFT",
            "Amazon.com, Inc. (AMZN)": "AMZN",
            "Custom Ticker": ""
        }
        selected_company = st.sidebar.selectbox("Select a Company", list(top_companies.keys()))

        if selected_company == "Custom Ticker":
            ticker = st.sidebar.text_input("Enter Stock Ticker")
        else:
            ticker = top_companies[selected_company]

        start_date = st.sidebar.date_input("Select Start Date", pd.to_datetime("2023-01-01"))
        end_date = st.sidebar.date_input("Select End Date", pd.to_datetime("2025-01-01"))

        if not ticker:
            st.warning("Please enter a valid stock ticker.")
            return

        st.write(f"Fetching data for {ticker}...")
        stock_data = self.fetch_stock_data(ticker, start_date, end_date)
        if stock_data is None or len(stock_data) < 60:
            st.error("Stock data not available or insufficient data. Please check the ticker or try again later.")
            return

        st.write(stock_data.head())

        # Prepare data
        X, y = self.prepare_lstm_data(stock_data)
        X_train, y_train = X[:int(0.8 * len(X))], y[:int(0.8 * len(y))]

        # Load or train model
        self.load_or_train_model(X_train, y_train)

        # Predict Future Prices
        future_price = self.predict_future_price(stock_data)
        if future_price:
            st.metric(label=f"📈 Predicted Price for {ticker} (Next Day)", value=f"${future_price:.2f}")

        # Sentiment Analysis (Example)
        example_news = "Stock market surges as investors remain optimistic about economic recovery."
        sentiment = self.analyze_sentiment(example_news)
        st.write(f"📰 Market Sentiment: {sentiment}")

        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["close"], mode="lines", name="Closing Price"))
        fig.update_layout(title=f"Stock Price Trend for {ticker}", xaxis_title="Date", yaxis_title="Price ($)")
        st.plotly_chart(fig)


# # Run the Streamlit application
# if __name__ == "__main__":
#     predictor = StockPredictor()
#     predictor.show_stock_prediction_page()
