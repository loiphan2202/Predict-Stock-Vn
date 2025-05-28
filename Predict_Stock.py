import datetime
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import requests
import seaborn as sns
import streamlit as st
import yfinance as yf
from keras.models import Sequential
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from function import *

plt.style.use("fivethirtyeight")
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stApp {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stSelectbox {
        background-color: white;
    }
    .stDateInput {
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1 style='color: #2E4053; font-size: 2.5rem;'>📈 Stock Price Predictor</h1>
        <p style='color: #566573; font-size: 1.2rem;'>Predict stock prices using LSTM algorithm</p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h2 style='color: #2E4053;'>Settings</h2>
        </div>
    """, unsafe_allow_html=True)
    
    today = datetime.datetime.today()
    YESTERDAY = today - BDay(0)
    DEFAULT_START = today - BDay(365)
    
    st.markdown("### Date Range")
    col1, col2 = st.columns(2)
    START = col1.date_input("From", value=DEFAULT_START, max_value=YESTERDAY)
    END = col2.date_input("To", value=YESTERDAY, max_value=YESTERDAY, min_value=START)
    
    st.markdown("### Stock Selection")
    STOCKS = np.array(["ACB", "BID", "CMC", "CMG", "CTG", "DXG", "FPT", "PVC",
                      "MSN", "PNJ", "PDR", "ELC", "KDH", "GAS", "VCB", "VIC", "VNM", "Another Choice"])
    SYMB = st.selectbox("Select stock", STOCKS)

if SYMB != "Another Choice":
    st.markdown(f"<h2 style='color: #2E4053; text-align: center;'>Price Data Analysis for {SYMB}</h2>", unsafe_allow_html=True)
    
    def get_stock_data_vndirect(symbol, start, end):
        start_dt = datetime.datetime.combine(start, datetime.datetime.min.time())
        end_dt = datetime.datetime.combine(end, datetime.datetime.min.time())

        start_ts = int(start_dt.timestamp())  
        end_ts = int(end_dt.timestamp())

        url = f"https://dchart-api.vndirect.com.vn/dchart/history?symbol={symbol}&resolution=D&from={start_ts}&to={end_ts}"
    
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if "t" not in data or not data["t"]:
                st.error(f"Lỗi: Không có dữ liệu từ API VNDIRECT cho {symbol}")
                return None
            df = pd.DataFrame({
                "time": pd.to_datetime(data["t"], unit="s"),
                "open": data["o"],
                "high": data["h"],
                "low": data["l"],
                "close": data["c"],
                "volume": data["v"]
            })
            df.set_index("time", inplace=True)
            return df
        else:
            st.error(f"Lỗi khi lấy dữ liệu từ VNDIRECT API (HTTP {response.status_code})")
            return None

    stock = get_stock_data_vndirect(SYMB, START, END)

    if stock is not None:
        with st.container():
            st.markdown("### Stock Data")
            st.dataframe(stock.style.background_gradient(cmap='RdYlGn'))

        field = np.array(["open", "high", "low", "close", "volume"])
        fd = st.selectbox("Select field to analyze", field)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Price Chart")
            xuatdothi_1(stock[fd])
        
        with col2:
            st.markdown("### Analysis")
            r_t, mean = fig_1(stock, fd, SYMB)
            fig_3(stock, fd, SYMB, r_t, mean)

        st.markdown("### Price Prediction")
        today = today - BDay(0)
        d30bf = today - BDay(100)
        data = get_stock_data_vndirect(SYMB, d30bf, today)
        data.reset_index(inplace=True)
        day = got_day()
        df_pred = predict(data, fd, SYMB, day)
        with st.container():
            st.markdown("""
                <div style='background-color: #fff; border-radius: 10px; padding: 1.5rem; box-shadow: 0 2px 8px #e0e0e0;'>
                    <h3 style='color: #2E4053;'>📊 Prediction Results</h3>
                </div>
            """, unsafe_allow_html=True)
            if df_pred is not None and not df_pred.empty:
                st.dataframe(df_pred)
            else:
                st.info("No prediction results available for the selected date.")
    else:
        st.error("Unable to fetch data from VNDIRECT API. Please check the stock symbol or try again later.")

else:
    st.markdown("### Upload Your Stock Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        stock = pd.read_csv(uploaded_file, index_col=0, parse_dates=True, infer_datetime_format=True)
        
        st.markdown("### Your Stock Data")
        st.dataframe(stock.style.background_gradient(cmap='RdYlGn'))
        
        sl = stock.columns
        fd = st.selectbox("Select field to analyze", sl)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Price Chart")
            xuatdothi_1(stock[fd])
        
        with col2:
            st.markdown("### Analysis")
            SYMB = 'Your_stock'
            r_t, mean = fig_1(stock, fd, SYMB)
            fig_3(stock, fd, SYMB, r_t, mean)

        st.markdown("### Price Prediction")
        today = datetime.datetime.today() - BDay(0)
        d30bf = today - BDay(100)
        day = got_day()
        df_pred = predict(stock, fd, SYMB, day)
        with st.container():
            st.markdown("""
                <div style='background-color: #fff; border-radius: 10px; padding: 1.5rem; box-shadow: 0 2px 8px #e0e0e0;'>
                    <h3 style='color: #2E4053;'>📊 Prediction Results</h3>
                </div>
            """, unsafe_allow_html=True)
            if df_pred is not None and not df_pred.empty:
                st.dataframe(df_pred)
            else:
                st.info("No prediction results available for the selected date.")
