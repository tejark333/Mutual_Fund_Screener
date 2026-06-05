import streamlit as st
import os
import pandas as pd


from analytics import *
from fund_analytics_page import *
from fund_compare_page import *
from portfolio_analyser_page import *

# if "active_tab" not in st.session_state:
#     st.session_state.active_tab = 0

PAGES = {
    "🧮 Fund Analytics": "fund_analytics",
    "🔁 Fund Compare": "fund_compare",
    "💼 Portfolio Analyzer": "portfolio"
}

with st.sidebar:
    st.title("📊 Mutual Fund Analytics")
    st.markdown("---")

    selected_page = st.radio(
        "Navigation",
        list(PAGES.keys()),
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("Built for analysis, not noise")

page = PAGES[selected_page]





# -----------------------------
# PAGE CONFIG
# -----------------------------
# st.set_page_config(
#     layout="wide",
#     page_title="Mutual Fund Analytics"
# )

# st.title("📊 Mutual Fund Analytics")


# def fund_compare_page():
#     st.header("🔁 Fund Compare")
#     st.info("Work in progress")

def portfolio_page():
    st.header("💼 Portfolio Analyzer")
    st.info("Coming soon")

def fund_compare_page_dummy():
    st.header("💼 Fund Comparison")
    st.info("Coming soon")

if page == "fund_analytics":
    # 🔹 EXISTING CODE GOES HERE (UNCHANGED)
    fund_analytics_page()

elif page == "fund_compare":
    fund_compare_page()

elif page == "portfolio":
    portfolio_analyser_page()
    

