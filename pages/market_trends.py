import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from textblob import TextBlob
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# Load environment variables for API keys
# load_dotenv("api_keys.env")
# # FMP_API_KEY = os.getenv("FMP_API_KEY")

if "FMP_API_KEY" in st.secrets:
    FMP_API_KEY = st.secrets["FMP_API_KEY"]
else:
    from dotenv import load_dotenv
    load_dotenv("api_keys.env")
    FMP_API_KEY = os.getenv("FMP_API_KEY")




# Load pre-trained LSTM model (Assuming model is already trained & saved)
try:
    model = load_model("stock_forecast_model.h5")
except Exception as e:
    model = None
    st.error(f"Could not load AI model: {e}")

# Portfolio Dictionary
user_portfolio = {}

# Fetch Market News
def fetch_market_news():
    url = f"https://financialmodelingprep.com/api/v3/stock_news?limit=10&apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data:
            return data
    return None

# Fetch Trending Stocks
def fetch_trending_stocks():
    url = f"https://financialmodelingprep.com/api/v3/stock_market/actives?apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    return None

# Fetch Historical Stock Data
def fetch_stock_history(symbol):
    url = f"https://financialmodelingprep.com/api/v3/historical-chart/5min/{symbol}?apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        df = pd.DataFrame(response.json())
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values("date", ascending=True, inplace=True)
        return df
    return None

# Sentiment Analysis on Stock News
def analyze_sentiment(news_text):
    sentiment_score = TextBlob(news_text).sentiment.polarity
    if sentiment_score > 0:
        return "📈 Positive"
    elif sentiment_score < 0:
        return "📉 Negative"
    else:
        return "⚖️ Neutral"

# Predict Future Stock Prices using LSTM
def predict_future_prices(stock_history):
    if model is None or stock_history is None:
        return None
    last_60_days = stock_history['close'].values[-60:].reshape(1, 60, 1)
    predicted_price = model.predict(last_60_days)
    return predicted_price[0][0]

# Add Stock to Portfolio
def add_to_portfolio(symbol, shares, purchase_price):
    user_portfolio[symbol] = {"Shares": shares, "Purchase Price": purchase_price}

# Show Portfolio
def show_portfolio():
    if user_portfolio:
        st.subheader("📊 My Portfolio")
        portfolio_df = pd.DataFrame(user_portfolio).T
        st.dataframe(portfolio_df)
    else:
        st.info("No stocks in portfolio. Add some to start tracking.")

# Display Market Trends & AI Analysis
def show_market_trends():
    st.title("📈 Market Trends & Insights 🔍")
    st.write("Stay updated with real-time stock market trends and financial insights!")
    st.markdown("---")

    trending_stocks = fetch_trending_stocks()
    
    if trending_stocks is not None and not trending_stocks.empty:
        trending_df = trending_stocks[['symbol', 'name', 'price', 'changesPercentage', 'change']]
        trending_df.columns = ["Ticker", "Company Name", "Current Price ($)", "Change (%)", "Change ($)"]

        col1, col2 = st.columns(2)

        with col1:
            st.write("### 📊 Stock Table")
            st.dataframe(trending_df, height=500)

        with col2:
            # Allow users to select a stock from dropdown OR enter a custom ticker
            selected_stock = st.selectbox("📊 Select a Stock from List", trending_df["Ticker"].tolist(), index=0)
            custom_stock = st.text_input("🔍 Or Enter a Custom Stock Ticker", value="")

            # Choose either the selected stock or custom input
            if custom_stock.strip():  
                ticker = custom_stock.upper()  # Use custom stock ticker if provided
            else:
                ticker = selected_stock  # Otherwise, use dropdown selection

            stock_history = fetch_stock_history(ticker)

            if stock_history is not None and not stock_history.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=stock_history["date"],
                    y=stock_history["close"],
                    mode="lines",
                    name=ticker,
                    line=dict(color='royalblue', width=2)
                ))
                fig.update_layout(
                    title=f"📊 {ticker} Live Stock Price Trends",
                    xaxis_title="Time",
                    yaxis_title="Price ($)",
                    template="plotly_white",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                future_price = predict_future_prices(stock_history)
                if future_price:
                    st.metric(label=f"📈 Predicted Price for {ticker} (Next Day)", value=f"${future_price:.2f}")

                shares = st.number_input("Enter number of shares to track", min_value=1, step=1)
                purchase_price = st.number_input("Enter purchase price per share ($)", min_value=0.01, step=0.01)
                if st.button("Add to Portfolio"):
                    add_to_portfolio(ticker, shares, purchase_price)
                    st.success(f"{shares} shares of {ticker} added to your portfolio!")
            else:
                st.warning(f"⚠️ Unable to fetch historical data for {ticker}. Please try again later.")
    else:
        st.error("⚠️ Unable to fetch trending stocks at the moment.")

    st.markdown("---")
    st.subheader("📊 Market Summary & Insights")
    st.write(
        """
        - 📌 **Track major indices like S&P 500, NASDAQ, and Dow Jones.**
        - 🚀 **Identify top gainers & losers in real-time.**
        - 🔍 **Analyze financial reports, earnings, and market movements.**
        - 🏆 **Get AI-powered insights on stock performance.**
        """
    )
    st.success("🚀 Stay ahead of the market! Check back regularly for live updates.")

    
        
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
                © 2025 StockMarketIQ. All rights reserved | Developed by Ajmal Amir.
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Call the footer function at the end of your main script
    add_footer()
    show_portfolio()

if __name__ == "__main__":
    show_market_trends()
