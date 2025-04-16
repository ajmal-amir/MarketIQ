import streamlit as st
import os
import pages as pg

# Initialize session state for page selection
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"  # Default page

# Define pages
pages = ["Home", "Stock Prediction", "Fnancial Analyst AI", "Market Trends", "User Guide"]

# --- Custom Navigation Bar ---
st.markdown(
    """
    <style>
    .navbar {
    }
    .nav-container {
        display: flex;
        justify-content: left;
        align-items: center;
        gap: 25px;
    }
    .nav-item button {
        background: none;
        color: #343a40;
        border: none;
        padding: 12px 20px;
        font-size: 16px;
        cursor: pointer;
        font-weight: 500;
        transition: color 0.3s;
    }
    .nav-item button:hover {
        color: black;
        font-weight: bold;
    }
    .selected button {
        color: black;
        font-weight: bold;
        text-decoration: underline;
    }
    </style>
    <div class="navbar">
        <div class="nav-container">
    """,
    unsafe_allow_html=True,
)

# Display buttons inside the navbar
cols = st.columns(len(pages))  # Create equal-width columns for each button
for i, page in enumerate(pages):
    with cols[i]:
        button_class = "selected" if st.session_state.selected_page == page else "nav-item"
        if st.button(page, key=page):
            st.session_state.selected_page = page

st.markdown("</div></div>", unsafe_allow_html=True)

# --- Page Content Handling ---
functions = {
    "Home": pg.show_home,
    "Market Trends": pg.show_market_trends,
    "Stock Prediction": pg.show_stock_predictionfmp,
    "Fnancial Analyst AI": pg.show_financial_analyst_ai,
    "User Guide": pg.show_user_guide
}

# Call the corresponding page function
if st.session_state.selected_page in functions:
    functions[st.session_state.selected_page]()
else:
    st.write("Page not found.")
