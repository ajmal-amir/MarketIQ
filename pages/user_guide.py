import streamlit as st



def show_user_guide():
    st.title("📘 StockMarketIQ - User Guide")
    st.subheader("🔹 Navigate the Stock Market with AI-Powered Insights!")
    
    st.markdown("---")

    st.markdown("## **🔹 Getting Started**")
    st.write(
        """
        Welcome to **StockMarketIQ**! Follow this guide to unlock the full potential of our AI-powered stock analysis and prediction tools.  
        Whether you're an **investor, trader, or financial analyst**, this platform equips you with cutting-edge insights!
        """
    )

    st.markdown("---")
    st.markdown("## **🛠 How to Use StockMarketIQ?**")
    
    st.markdown("### **1️⃣ Home Page - Overview & Navigation**")
    st.write(
        """
        - This is your **dashboard**, providing an overview of what StockMarketIQ offers.  
        - You’ll find **three main sections** to explore:
            1. **Stock Prediction App 📊** – AI-powered stock forecasting.  
            2. **Stock Analysis AI 📈** – In-depth market analysis using AI.  
            3. **Market Trends & Insights 🔍** – Stay updated with real-time trends.  
        - Use the **navigation buttons** to switch between different features easily.
        """
    )

    st.markdown("### **2️⃣ Selecting a Stock**")
    st.write(
        """
        - Choose from the **Top 20 Market Leaders** or enter a **custom stock ticker**.  
        - Enter your preferred **stock symbol (e.g., AAPL, TSLA, MSFT)** in the input box.  
        - Our system will fetch **real-time and historical data** for analysis.  
        """
    )

    st.markdown("### **3️⃣ Setting the Date Range**")
    st.write(
        """
        - Use the **date selection tool** to define the timeframe for prediction and analysis.  
        - Choose a **start date** and an **end date** to customize the analysis period.  
        - The AI model will train on historical data within this range.  
        """
    )

    st.markdown("### **4️⃣ AI-Powered Stock Predictions**")
    st.write(
        """
        - Select from **multiple AI & ML models** for stock forecasting:
            - ✅ **Artificial Neural Networks (ANN)**
            - ✅ **K-Nearest Neighbors (k-NN)**
            - ✅ **Support Vector Machine (SVM)**
            - ✅ **Decision Tree (C4.5)**
            - ✅ **Random Forest (RF)**
            - ✅ **Bagging & Boosting Models (AdaBoost, Bagging)**  
        - The AI will process the data and **predict future stock movements**.  
        """
    )

    st.markdown("### **5️⃣ Visualizing the Predictions**")
    st.write(
        """
        - Select a **graph type** to display predictions:
            - 📉 **Line Graph** – Shows trends over time.  
            - 📊 **Bar Graph** – Compares actual vs predicted values.  
            - ⚫ **Scatter Plot** – Displays individual data points.  
        - The chart will be generated **automatically** once predictions are ready.  
        """
    )

    st.markdown("### **6️⃣ Understanding Model Accuracy**")
    st.write(
        """
        - For each model, you’ll get **accuracy metrics**:
            - **Mean Absolute Error (MAE)** – Measures prediction error.  
            - **Mean Squared Error (MSE)** – Evaluates error spread.  
            - **R² Score** – Determines how well predictions match real data.  
        - The **lower** the MAE and MSE, the **better the model's accuracy**!  
        """
    )

    st.markdown("### **7️⃣ Customizing Stock Analysis**")
    st.write(
        """
        - Use **real-time AI analysis** to gain insights into stock trends.  
        - Apply **filters** to refine results based on volume, volatility, and market patterns.  
        - Export analysis results for **further review** or **reporting**.  
        """
    )

    st.markdown("---")
    st.markdown("## **💡 Pro Tips for Better Results**")
    st.write(
        """
        - 📌 **Choose a wider date range** for better model training.  
        - 📊 **Compare multiple models** before finalizing predictions.  
        - 🚀 **Use the most recent data** for up-to-date stock analysis.  
        - 🔍 **Test different stocks** to understand AI forecasting power.  
        """
    )

    st.markdown("---")
    st.markdown("## **❓ Need Help?**")
    st.write(
        """
        - Click the **Chat Icon 💬** to connect with a support agent.  
        - Visit the **FAQ Section 📚** for common troubleshooting tips.  
        - Reach out via **support@stockmarketiq.com** for further assistance.  
        """
    )

    st.success("🎉 **You're now ready to explore StockMarketIQ! Start making data-driven investment decisions today!** 🚀")


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

