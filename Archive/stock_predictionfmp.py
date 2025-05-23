import streamlit as st

def show_stock_predictionfmp():
    st.header("Stock Prediction")
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import requests
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from dotenv import load_dotenv
    from pages.StockPredictor import StockPredictor
    import plotly.graph_objects as go


    #stock predicore class 
    stock_predictor = StockPredictor()
    
    # Function to fetch stock data from FMP API
    def fetch_stock_data(symbol, start_date, end_date, api_key):
        base_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
        params = {
            "from": start_date,
            "to": end_date,
            "apikey": api_key
        }
        response = requests.get(base_url, params=params)
        data = response.json()
        if "historical" in data:
            df = pd.DataFrame(data["historical"])
            df = df[["date", "open", "high", "low", "close", "volume"]]
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            return df
        else:
            return pd.DataFrame()
    
  
    
    
    st.title("Stock Market Prediction Using Machine Learning")
    st.sidebar.header("User Input Parameters")
    
    # Top 20 U.S. companies by market capitalization
    top_companies = {
        "Apple Inc. (AAPL)": "AAPL",
        "Microsoft Corporation (MSFT)": "MSFT",
        "Amazon.com, Inc. (AMZN)": "AMZN",
        "Alphabet Inc. Class A (GOOGL)": "GOOGL",
        "Alphabet Inc. Class C (GOOG)": "GOOG",
        "Berkshire Hathaway Inc. (BRK.B)": "BRK.B",
        "NVIDIA Corporation (NVDA)": "NVDA",
        "Meta Platforms, Inc. (META)": "META",
        "Tesla, Inc. (TSLA)": "TSLA",
        "Visa Inc. (V)": "V",
        "Johnson & Johnson (JNJ)": "JNJ",
        "Walmart Inc. (WMT)": "WMT",
        "Procter & Gamble Co. (PG)": "PG",
        "Mastercard Incorporated (MA)": "MA",
        "UnitedHealth Group Incorporated (UNH)": "UNH",
        "The Home Depot, Inc. (HD)": "HD",
        "Samsung Electronics Co., Ltd. (SSNLF)": "SSNLF",
        "Kweichow Moutai Co., Ltd. (600519.SS)": "600519.SS",
        "Roche Holding AG (RHHBY)": "RHHBY",
        "Alibaba Group Holding Limited (BABA)": "BABA",
        "Custom Ticker": ""
    }
    # Dropdown menu for top companies
    selected_company = st.sidebar.selectbox("Select a Company", list(top_companies.keys()))
    
    # If 'Custom Ticker' is selected, show text input for custom ticker
    if selected_company == "Custom Ticker":
        ticker = st.sidebar.text_input("Enter Stock Ticker")
    else:
        ticker = top_companies[selected_company]
    
                                                                     
    
    # ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
    start_date = st.sidebar.date_input("Select Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("Select End Date", pd.to_datetime("2025-03-30"))
    prediction_days = st.sidebar.slider("Prediction Duration (Days)", 1, 60, 30)
    
    graph_options = ["Line Graph", "Bar Graph", "Scatter Plot"]
    selected_graph = st.sidebar.selectbox("Choose a Graph Type", graph_options)
    
    # Ensure you specify the correct .env file name
    load_dotenv("api_keys.env")
    
    # Retrieve API key from environment variable
    api_key = os.getenv("FMP_API_KEY")
    
    if not api_key:
        st.error("API key not found. Please ensure it's set in the .env file.")
        st.stop()
    
    st.write(f"Fetching historical data for {ticker}...")
    stock_data = fetch_stock_data(ticker, start_date, end_date, api_key)
    if stock_data.empty:
        st.error("Stock data is empty. Please check the ticker symbol or date range.")
        st.stop()

    # Display Historical Data Table & LSTM Prediction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Historical Stock Data")
        st.dataframe(stock_data.tail(10))  # Display last 10 days
    
    with col2:
        future_price = stock_predictor.predict_future_price(stock_data)
        if future_price:
            st.metric(label=f"📈 Predicted Price for {selected_company} ({ticker}) (Next Day)", value=f"${future_price:.2f}")
            
    
    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["close"], mode="lines", name="Closing Price"))
    fig.update_layout(title=f"Stock Price Trend for {ticker}", xaxis_title="Date", yaxis_title="Price ($)")
    st.plotly_chart(fig)
    
    # st.write(stock_data.head())
    
    # Data Preprocessing
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_data)
    X = scaled_data[:, :-1]
    y = scaled_data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and Test Multiple Models
    models = {
        "Artificial Neural Network (ANN)": Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1)
        ]),
        "K-Nearest Neighbor (k-NN)": KNeighborsRegressor(n_neighbors=5),
        "Support Vector Machine (SVM)": SVR(kernel='rbf'),
        "Decision Tree (C4.5)": DecisionTreeRegressor(),
        "Random Forest (RF)": RandomForestRegressor(n_estimators=100),
        "Bagging": BaggingRegressor(n_estimators=50),
        "AdaBoost": AdaBoostRegressor(n_estimators=50)
    }
    
    predictions = {}
    accuracies = {}
    for model_name, model in models.items():
        st.write(f"Training {model_name}...")
        if model_name == "Artificial Neural Network (ANN)":
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        else:
            model.fit(X_train, y_train)
        pred = model.predict(X_test).flatten()
        predictions[model_name] = pred
        
        # Calculate accuracy metrics
        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        accuracies[model_name] = {
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "R2 Score": round(r2, 4)
        }
    
    # Plot Results for Each Model Separately
    # Plot Results for Each Model Separately
    for model_name, pred in predictions.items():
        st.write(f"### {model_name} Predictions vs Actual")
        st.write("**Model Accuracy:**")
        st.write(f"- MAE: {accuracies[model_name]['MAE']}")
        st.write(f"- MSE: {accuracies[model_name]['MSE']}")
        st.write(f"- R2 Score: {accuracies[model_name]['R2 Score']}")
        
        fig, ax = plt.subplots()
        days = range(len(y_test))
        if selected_graph == "Line Graph":
            ax.plot(days, y_test, label='Actual', color='black')
            ax.plot(days, pred, label=f'{model_name} Predictions', linestyle='dashed', color='red')
        elif selected_graph == "Bar Graph":
            ax.bar(days, y_test, label='Actual', color='black', alpha=0.6)
            ax.bar(days, pred, label=f'{model_name} Predictions', alpha=0.6, color='red')
        elif selected_graph == "Scatter Plot":
            ax.scatter(days, y_test, label='Actual', color='black')
            ax.scatter(days, pred, label=f'{model_name} Predictions', color='red')
        
        ax.set_xlabel("Days")
        ax.set_ylabel("Stock Price")
        ax.set_title(f"{model_name} Prediction Performance over {len(y_test)} days")
        ax.legend()
        st.pyplot(fig)
    
    
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
    
    # Call the footer function at the end of your main script
    add_footer()

