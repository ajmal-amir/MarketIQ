
def show_financial_analyst_ai():
    import streamlit as st
    st.header("Fnancial Analyst AI")
    import requests
    import pandas as pd
    import numpy as np
   
    import os
    import matplotlib.pyplot as plt
    from typing import Dict, TypedDict, Optional
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq
    from dotenv import load_dotenv
    
    # Load environment variables
    # Load environment variables
    load_dotenv("API_Keys.env")  # Ensure this file is in the same directory
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if not GROQ_API_KEY:
        st.error("Error: GROQ_API_KEY is not set. Please check your .env file.")
    FMP_API_KEY = os.getenv("FMP_API_KEY")
    
    # Define State Structure
    class State(TypedDict):
        stock_symbol: str
        stock_data: Optional[pd.DataFrame]
        indicators: Optional[Dict[str, float]]
        financial_metrics: Optional[Dict[str, float]]
        analysis: Optional[str]
        summary: Optional[str]
        approved: Optional[bool]
    
    # Fetch stock data from FMP
    def fetch_stock_data(state: State) -> State:
        symbol = state["stock_symbol"].upper()
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={FMP_API_KEY}"
        
        response = requests.get(url)
        
        if response.status_code != 200:
            st.error(f"Error fetching stock data: {response.text}")
            return state
        
        data = response.json()
        
        if "historical" in data and isinstance(data["historical"], list) and len(data["historical"]) > 0:
            hist = pd.DataFrame(data["historical"])
            hist = hist.sort_values(by="date", ascending=True)
            hist.set_index("date", inplace=True)
            state["stock_data"] = hist
        else:
            st.warning(f"No data found for symbol {symbol}. Please check the symbol.")
            state["stock_data"] = None
    
        return state
    
    # Compute technical indicators
    def compute_indicators(state: State) -> State:
        data = state["stock_data"]
        indicators = {}
    
        if data is None or data.empty:
            return state
    
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
    
        indicators['RSI'] = 100 - (100 / (1 + rs.iloc[-1])) if not rs.isnull().all() else None
    
        short_ema = data['close'].ewm(span=12, adjust=False).mean()
        long_ema = data['close'].ewm(span=26, adjust=False).mean()
        macd = short_ema - long_ema
    
        indicators['MACD'] = macd.iloc[-1] if not macd.isnull().all() else None
    
        state["indicators"] = indicators
        return state
    
    # Fetch financial metrics from FMP
    def compute_financial_metrics(state: State) -> State:
        symbol = state["stock_symbol"]
        url = f"https://financialmodelingprep.com/api/v3/ratios/{symbol}?apikey={FMP_API_KEY}"
        
        response = requests.get(url)
        
        if response.status_code != 200:
            st.error(f"Error fetching financial metrics: {response.text}")
            return state
    
        data = response.json()
        
        if isinstance(data, list) and len(data) > 0:
            latest_metrics = data[0]
            state["financial_metrics"] = {
                "P/E Ratio": latest_metrics.get("priceEarningsRatio", None),
                "Debt-to-Equity": latest_metrics.get("debtEquityRatio", None),
                "Profit Margin": latest_metrics.get("netProfitMargin", None)
            }
        else:
            st.warning(f"No financial metrics found for {symbol}.")
            state["financial_metrics"] = None
    
        return state
    
    # Generate AI-driven financial analysis
    def generate_analysis(state: State) -> State:
        if not state.get("indicators") or not state.get("financial_metrics"):
            state["analysis"] = "Not enough data to generate analysis."
            return state
    
        prompt = ChatPromptTemplate.from_template(
            """
            Given the following stock data:
            
            Stock Symbol: {stock_symbol}
            Indicators: {indicators}
            Financial Metrics: {financial_metrics}
            
            Provide an AI-driven analysis of this stock's performance and 
            investment potential. Explain each metric and its impact. Be concise 
            and structured, mentioning risks and opportunities.
            """
        )
        model = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-70b-8192")
        response = model.invoke(prompt.format(
            stock_symbol=state["stock_symbol"],
            indicators=state["indicators"],
            financial_metrics=state["financial_metrics"]
        ))
        state["analysis"] = response.content
        return state
    
    # Generate AI Summary Report
    def generate_summary(state: State) -> State:
        if not state.get("analysis"):
            state["summary"] = "No summary available."
            return state
    
        prompt = ChatPromptTemplate.from_template(
            """
            Given this AI-generated stock analysis:
            
            {analysis}
            
            Provide a concise and user-friendly summary highlighting the key takeaways.
            Make it clear for non-experts what they should know about this stock.
            """
        )
        model = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-70b-8192")
        response = model.invoke(prompt.format(analysis=state["analysis"]))
        state["summary"] = response.content
        return state
    
    # Initialize session state
    if "workflow_state" not in st.session_state:
        st.session_state["workflow_state"] = {}
    
    # Input for stock symbol
    stock_symbol = st.text_input("Enter Stock Symbol:", st.session_state["workflow_state"].get("stock_symbol", "MSFT"))
    
    # **Reset state every time stock ticker changes**
    if stock_symbol != st.session_state["workflow_state"].get("stock_symbol", ""):
        st.session_state["workflow_state"] = {
            "stock_symbol": stock_symbol,
            "approved": False,  # Reset approval every time
            "stock_data": None,
            "indicators": None,
            "financial_metrics": None,
            "analysis": None,
            "summary": None
        }
    
    # Function to plot RSI with user-selected graph type
    def plot_rsi(data, graph_type):
        if data is not None:
            fig, ax = plt.subplots()
            data["RSI"] = 100 - (100 / (1 + (data['close'].diff().where(lambda x: x > 0, 0).rolling(14).mean() / 
                                             data['close'].diff().where(lambda x: x < 0, 0).rolling(14).mean().abs())))
    
            if graph_type == "Line Plot":
                ax.plot(data.index, data["RSI"], label="RSI", color="blue")
            elif graph_type == "Bar Chart":
                ax.bar(data.index, data["RSI"], label="RSI", color="blue", alpha=0.6)
            elif graph_type == "Scatter Plot":
                ax.scatter(data.index, data["RSI"], label="RSI", color="blue")
    
            ax.axhline(70, linestyle="--", color="red", label="Overbought (70)")
            ax.axhline(30, linestyle="--", color="green", label="Oversold (30)")
            ax.set_title("RSI (Relative Strength Index)")
            ax.legend()
            st.pyplot(fig)
    
    # Function to plot MACD with user-selected graph type
    def plot_macd(data, graph_type):
        if data is not None:
            short_ema = data['close'].ewm(span=12, adjust=False).mean()
            long_ema = data['close'].ewm(span=26, adjust=False).mean()
            macd = short_ema - long_ema
            signal = macd.ewm(span=9, adjust=False).mean()
    
            fig, ax = plt.subplots()
    
            if graph_type == "Line Plot":
                ax.plot(data.index, macd, label="MACD", color="blue")
                ax.plot(data.index, signal, label="Signal Line", color="red")
            elif graph_type == "Bar Chart":
                ax.bar(data.index, macd, label="MACD", color="blue", alpha=0.6)
                ax.bar(data.index, signal, label="Signal Line", color="red", alpha=0.6)
            elif graph_type == "Scatter Plot":
                ax.scatter(data.index, macd, label="MACD", color="blue")
                ax.scatter(data.index, signal, label="Signal Line", color="red")
    
            ax.set_title("MACD (Moving Average Convergence Divergence)")
            ax.legend()
            st.pyplot(fig)
    
    
    
    
    # Button to analyze stock
    if st.button("Analyze Stock"):
        with st.spinner("Fetching stock data..."):
            state = fetch_stock_data(st.session_state["workflow_state"])
            state = compute_indicators(state)
            state = compute_financial_metrics(state)
            state = generate_analysis(state)  # AI Analysis BEFORE Approval
            st.session_state["workflow_state"] = state  # Update session state
    
            
    
    # Display stock data
    if st.session_state["workflow_state"].get("stock_data") is not None:
        st.subheader("Stock Data")
        st.dataframe(st.session_state["workflow_state"]["stock_data"].tail())
    
    # Display technical indicators
    if st.session_state["workflow_state"].get("indicators"):
        st.subheader("Technical Indicators")
        for key, value in st.session_state["workflow_state"]["indicators"].items():
            st.write(f"{key}: {value}")
        # User selection for RSI and MACD graph types
        rsi_graph_type = st.selectbox("Select RSI Graph Type:", ["Line Plot", "Bar Chart", "Scatter Plot"])
        
        
        plot_rsi(st.session_state["workflow_state"]["stock_data"], rsi_graph_type)
        
    
    # Display financial metrics
    if st.session_state["workflow_state"].get("financial_metrics"):
        st.subheader("Financial Metrics")
        for key, value in st.session_state["workflow_state"]["financial_metrics"].items():
            st.write(f"{key}: {value}")
            
        macd_graph_type = st.selectbox("Select MACD Graph Type:", ["Line Plot", "Bar Chart", "Scatter Plot"])
        plot_macd(st.session_state["workflow_state"]["stock_data"], macd_graph_type)
    
    
    # Display AI-generated analysis **before approval**
    if st.session_state["workflow_state"].get("analysis"):
        st.subheader("AI-Generated Analysis")
        st.write(st.session_state["workflow_state"]["analysis"])
    
    # **Approval step happens AFTER AI analysis**
    st.subheader("Human Approval")
    approval = st.radio("Approve the AI analysis?", ["Pending", "Approved"], index=0, key=f"approval_{stock_symbol}")
    
    # **Ensure approval resets every time before generating summary**
    if approval == "Approved" and not st.session_state["workflow_state"].get("approved"):
        st.session_state["workflow_state"]["approved"] = True
        st.session_state["workflow_state"]["summary"] = None  # Reset summary for fresh generation
    
    
    
    # **Generate AI summary if approved**
    if st.session_state["workflow_state"]["approved"] and not st.session_state["workflow_state"].get("summary"):
        with st.spinner("Generating Summary Report..."):
            state = generate_summary(st.session_state["workflow_state"])
            st.session_state["workflow_state"]["summary"] = state["summary"]
    
    
    # Display AI-generated Summary
    if st.session_state["workflow_state"]["approved"]:
        st.success("Approval Complete. Here is your summary:")
        st.subheader("AI-Generated Summary Report")
        st.write(st.session_state["workflow_state"]["summary"])

        
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
