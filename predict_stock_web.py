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
# from train import train

plt.style.use("fivethirtyeight")
warnings.filterwarnings("ignore")



st.set_page_config(layout="wide", initial_sidebar_state="expanded")
# st.image("",width = 150) #logo
# st.markdown('') # brand name

   
# ------ layout setting---------------------------
window_selection_c = st.sidebar.container() # create an empty container in the sidebar
# window_selection_c.markdown("") # add a title to the sidebar container
sub_columns = window_selection_c.columns(2) #Split the container into two columns for start and end date

today = datetime.datetime.today()
YESTERDAY = today - BDay(0)

DEFAULT_START=today -  BDay(365)
START = sub_columns[0].date_input("From", value=DEFAULT_START, max_value=YESTERDAY)
END = sub_columns[1].date_input("To", value=YESTERDAY, max_value=YESTERDAY, min_value=START)


STOCKS = np.array([ "ACB", "BID", "CMC", "CMG", "CTG", "DXG", "FPT", "PVC",
                     "MSN", "PNJ", "PDR", "ELC", "KDH", "GAS", "VCB", "VIC", "VNM","Another Choice"])
SYMB = window_selection_c.selectbox("select stock", STOCKS)

if SYMB != "Another Choice":
# # # # ------------------------Plot stock linecharts--------------------
    st.title('Price data of '+SYMB+' stock')
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
        st.write(stock)

        if stock is not None and not stock.empty:
            field = np.array(["open", "high", "low", "close", "volume"])
            fd = window_selection_c.selectbox("Select field", field)

            xuatdothi_1(stock[fd])  # Bây giờ sẽ hoạt động vì tên cột khớp với dữ liệu
        else:
            st.error("Không có dữ liệu hợp lệ, vui lòng kiểm tra lại mã cổ phiếu hoặc khoảng thời gian.")
    
    # Chỉ thực hiện drop khi stock không phải None
        if stock is not None:
            stock = stock.drop(columns=['Stock Splits'], errors='ignore')  # Thêm `errors='ignore'` để tránh lỗi nếu không có cột 'Stock Splits'
            stock = stock.drop(columns=['Dividends'], errors='ignore')  # Tương tự với cột 'Dividends'
            st.write(stock)

    # Tiếp tục vẽ đồ thị sau khi xử lý
        r_t, mean = fig_1(stock, fd, SYMB)
        fig_3(stock, fd, SYMB, r_t, mean)
    else:
        st.error("Không có dữ liệu từ VNDIRECT API hoặc không thể truy cập vào dữ liệu.")



    # #----part-1--------------------------------Session state intializations---------------------------------------------------------------

    if "TEST_INTERVAL_LENGTH" not in st.session_state:
        # set the initial default value of test interval
        st.session_state.TEST_INTERVAL_LENGTH = 60

    if "TRAIN_INTERVAL_LENGTH" not in st.session_state:
        # set the initial default value of the training length widget
        st.session_state.TRAIN_INTERVAL_LENGTH = 500

    if "HORIZON" not in st.session_state:
        # set the initial default value of horizon length widget
        st.session_state.HORIZON = 60

    if 'TRAINED' not in st.session_state:
        st.session_state.TRAINED=False

    # #---------------------------------------------------------Train_test_forecast_splits---------------------------------------------------
    today = today - BDay(0)
    d30bf = today - BDay(100)
    data = get_stock_data_vndirect(SYMB, d30bf, today)
    data.reset_index(inplace=True)
    day = got_day()
    predict(data,fd,SYMB,day)
    # THêm bảng testing error ss dữ liệu test vs dự đoán
else:
    uploaded_file = window_selection_c.file_uploader("Choose a file")
    stock = pd.DataFrame()
    if uploaded_file is not None:
        stock = pd.read_csv(uploaded_file,index_col=0,parse_dates=True,infer_datetime_format=True)
        # print(stock)
        
        sl = stock.columns
        fd = window_selection_c.selectbox("select field to show", sl)
        st.title('Price data of your stock')
        xuatdothi_1(stock[fd])
        st.write(stock)
        SYMB = 'Your_stock'
        r_t,mean = fig_1(stock,fd,SYMB)
        fig_3(stock,fd,SYMB,r_t,mean)
        Size = stock[fd].shape[0]
        stock.reset_index(inplace=True)
        # Button = window_selection_c.button("Train")
        # if Button:
        #     for ld in sl:
        #         train(stock,ld)
        # day = got_day()
        
        # fdd = window_selection_c.selectbox("select field to predict",sl)
        
        # predict(stock[:30],fd,'Your_stock',day)
        