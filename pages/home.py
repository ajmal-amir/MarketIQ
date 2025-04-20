import streamlit as st

st.set_page_config(page_title="home", layout="wide")

# Function to show disclaimer popup
def show_disclaimer():
    with st.expander("⚠️ **Disclaimer - Please Read Before Proceeding** ⚠️", expanded=True):
        st.write(
            """
            **StockMarketIQ is a course assignment and should not be considered as professional financial advice.**  
            The information, predictions, and market insights provided in this application are for **educational purposes only**.

            ### **Important Notes:**
            - 📌 **No investment guarantees** – Stock predictions are based on **historical data and AI models**, which **do not guarantee future performance**.
            - 📊 **Third-party data sources** – Market trends, financial news, and stock prices are **retrieved from external APIs**, and we **do not guarantee their accuracy or availability**.
            - 🔍 **No liability** – The creators of this application **are not responsible** for any financial losses or decisions made based on the information provided.
            - 🚀 **Always consult a financial advisor** before making investment decisions.

            **By using this application, you acknowledge that StockMarketIQ is a student project and not a professional trading or investment platform.**  
            """
        )
        st.success("✅ Click outside this message to continue.")

# Footer function (properly aligned now)
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
            © 2025 StockMarketIQ. All rights reserved. | Developed by Ajmal Amir
        </div>
        """,
        unsafe_allow_html=True,
    )

# Function to show the Home Page
def show_home():
    show_disclaimer()  # Show disclaimer popup when the Home Page loads

    st.title("Welcome to StockMarketIQ 📈💡")
    st.subheader("Your AI-Powered Stock Market Companion!")

    st.write(
        """
        🚀 **StockMarketIQ** is an advanced AI-driven platform designed for **investors, traders, and financial analysts**.  
        Our powerful machine learning models provide **real-time stock predictions** and **deep market insights** to help you make **data-driven investment decisions**.
        """
    )

    st.markdown("### **🌟 Why Choose StockMarketIQ?**")
    st.write(
        """
        ✅ **AI-Powered Stock Predictions** – Forecast stock prices using top AI algorithms.  
        ✅ **Real-Time Market Insights** – Stay ahead of trends with AI-driven stock analysis.  
        ✅ **Intuitive Data Visualization** – View insights through interactive charts and reports.  
        ✅ **Top Market Leaders & Custom Tickers** – Track major companies or enter your own stock ticker.  
        ✅ **User-Friendly Interface** – Navigate easily with our intuitive design.  
        """
    )

    add_footer()  #Call the footer here

# Example usage:
if __name__ == "__main__":
    show_home()
