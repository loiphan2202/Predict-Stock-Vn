import base64
import datetime
import io
import logging
import os
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from keras.models import Sequential
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

app = Flask(__name__)
CORS(app)
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Hamf chuyeenr đổi định dang dữ liệuliệu
def got_data(data):
    input = list()
    input.append(data[:])
    return np.asarray(input)

def xuly(data, SYMB, fd):
    try:
        if len(data) < 30:
            logger.error(f"Không đủ dữ liệu đầu vào (cần ít nhất 30 điểm dữ liệu, chỉ có {len(data)})")
            return None
            
        input = got_data(data)
        scaler = MinMaxScaler()
        scaler.fit_transform(input)
        input = input.reshape(-1, 30, 1)
        
        model_path = f'Model/{SYMB}_{fd}.h5'
        model = None
        
        if not os.path.exists(model_path):
            logger.info(f"Tạo mô hình mới cho {SYMB}_{fd}")
            model = Sequential()
            model.add(keras.layers.LSTM(50, return_sequences=True, input_shape=(30, 1)))
            model.add(keras.layers.LSTM(50, return_sequences=False))
            model.add(keras.layers.Dense(25))
            model.add(keras.layers.Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            model.save(model_path)
        else:
            try:
                logger.info(f"Đang tải mô hình từ {model_path}")
                model = tf.keras.models.load_model(model_path, compile=False)
            except Exception as e:
                logger.error(f"Lỗi khi tải mô hình: {str(e)}")
                model = Sequential()
                model.add(keras.layers.LSTM(50, return_sequences=True, input_shape=(30, 1)))
                model.add(keras.layers.LSTM(50, return_sequences=False))
                model.add(keras.layers.Dense(25))
                model.add(keras.layers.Dense(1))
                
                try:
                    with h5py.File(model_path, 'r') as f:
                        weight_layers = [layer for layer in model.layers if len(layer.weights) > 0]
                        
                        for i, layer in enumerate(weight_layers):
                            if f'layer_{i}' in f:
                                weights = []
                                for j in range(len(layer.weights)):
                                    if f'layer_{i}/weight_{j}' in f:
                                        weights.append(f[f'layer_{i}/weight_{j}'][()])
                                
                                if weights:
                                    layer.set_weights(weights)
                    
                    model.save(model_path)
                except Exception as e:
                    logger.error(f"Lỗi khi đọc trọng số: {str(e)}")
                    model = Sequential()
                    model.add(keras.layers.LSTM(50, return_sequences=True, input_shape=(30, 1)))
                    model.add(keras.layers.LSTM(50, return_sequences=False))
                    model.add(keras.layers.Dense(25))
                    model.add(keras.layers.Dense(1))
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    model.save(model_path)
        
        if model is None:
            logger.error("Không thể tạo hoặc tải mô hình")
            return None
            
        # Dự đoán với mô hình
        testk = model.predict(input, verbose=0)
        m = float(testk)
        mn = min(scaler.data_min_)
        e = (max(scaler.data_max_) - min(scaler.data_min_))
        return m * e + mn
    except Exception as e:
        logger.error(f"Lỗi trong quá trình xử lý dự đoán: {str(e)}")
        return None

def get_stock_data_vndirect(symbol, start, end):
    try:
        start_dt = datetime.datetime.combine(start, datetime.datetime.min.time())
        end_dt = datetime.datetime.combine(end, datetime.datetime.min.time())

        start_ts = int(start_dt.timestamp())  
        end_ts = int(end_dt.timestamp())

        url = f"https://dchart-api.vndirect.com.vn/dchart/history?symbol={symbol}&resolution=D&from={start_ts}&to={end_ts}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        
        logger.info(f"Đang lấy dữ liệu cho {symbol} từ {start} đến {end}")
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Lỗi API ({response.status_code}): {response.text}")
            return None

        try:
            data = response.json()
        except ValueError:
            logger.error(f"Không thể chuyển response thành JSON: {response.text}")
            return None

        if "t" not in data or not data["t"]:
            logger.error(f"Không có dữ liệu thời gian trong phản hồi: {data}")
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

    except requests.RequestException as e:
        logger.error(f"Lỗi kết nối đến API: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Lỗi không mong đợi: {str(e)}")
        return None

def predict(data, fd, SYMB, day):
    try:
        today = datetime.datetime.today()
        today = today - BDay(0)
        
        if day < today.date():
            logger.warning(f"Ngày dự đoán {day} nằm trong quá khứ")
        
        if fd not in data.columns:
            logger.error(f"Trường {fd} không tồn tại trong dữ liệu")
            return {"error": f"Trường {fd} không tồn tại trong dữ liệu"}
        
        data_series = data[:][fd]
        
        if len(data_series) < 30:
            logger.error(f"Không đủ dữ liệu đầu vào (cần ít nhất 30 điểm dữ liệu, chỉ có {len(data_series)})")
            return {"error": "Không đủ dữ liệu đầu vào (cần ít nhất 30 điểm dữ liệu)"}
            
        data_series = data_series[-30:]
        L = list(data_series)
        
        date1 = list()
        ketqua = list()
        mix = {}
        
        prediction_day = datetime.datetime.combine(day, datetime.datetime.min.time())
        forecast_end = prediction_day if prediction_day > today else today
        
        current = today
        while current <= forecast_end:
            date1.append(current)
            x = xuly(np.array(L), SYMB, fd)
            
            if x is None:
                logger.error("Không thể dự đoán giá trị")
                return {"error": "Không thể dự đoán giá trị"}
                
            L = L[1:] + [x]
            ketqua.append(x)
            mix[current.strftime('%Y-%m-%d')] = x
            current = current - BDay(-1)
            
        return {"dates": [d.strftime('%Y-%m-%d') for d in date1], "values": ketqua, "prediction_data": mix}
    except Exception as e:
        logger.error(f"Lỗi trong quá trình dự đoán: {str(e)}")
        return {"error": f"Lỗi trong quá trình dự đoán: {str(e)}"}

def create_price_movement_chart(stock, fd, SYMB):
    try:
        if stock is None or fd not in stock.columns:
            return {"error": "Dữ liệu không hợp lệ"}
            
        r_t = np.log((stock[fd]/stock[fd].shift(1)))
        r_t = r_t.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(r_t) == 0:
            return {"error": "Không thể tính toán tỷ lệ lợi nhuận"}
            
        mean = np.mean(r_t)
        
        if pd.isna(r_t.iloc[0]):
            r_t.iloc[0] = mean
        
        plt.figure(figsize=(12, 5))
        plt.plot(r_t.index, r_t.values, linestyle='--', marker='o')
        plt.axhline(y=mean, label='mean return', c='red')
        plt.legend()
        plt.title(f'Price movement chart {fd} of {SYMB}')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return {"image": img_str, "mean_return": float(mean), "return_rates": r_t.tolist()}
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ chuyển động giá: {str(e)}")
        return {"error": f"Lỗi khi tạo biểu đồ: {str(e)}"}

def create_avg_profit_chart(r_t, mean, SYMB):
    try:
        plt.figure(figsize=(12, 5))

        sns.histplot(r_t, kde=True, bins=20)
        plt.axvline(x=mean, label='mean return', c='red')
        plt.legend()
        plt.xlabel('return rate')
        plt.ylabel('frequency')
        plt.title(f'Overall average daily profit of {SYMB}')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return {"image": img_str}
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ lợi nhuận: {str(e)}")
        return {"error": f"Lỗi khi tạo biểu đồ: {str(e)}"}

@app.route('/', methods=['GET'])
def api_documentation():
    return jsonify({
        "api_name": "Stock Price Prediction API",
        "endpoints": [
            {"path": "/api/stocks", "method": "GET", "description": "Lấy danh sách mã cổ phiếu có sẵn"},
            {"path": "/api/stock-data", "method": "GET", "params": ["symbol", "start", "end"], "description": "Lấy dữ liệu cổ phiếu trong khoảng thời gian"},
            {"path": "/api/price-movement", "method": "GET", "params": ["symbol", "field", "start", "end"], "description": "Tạo biểu đồ chuyển động giá"},
            {"path": "/api/avg-profit", "method": "GET", "params": ["symbol", "field", "start", "end"], "description": "Tạo biểu đồ lợi nhuận trung bình"},
            {"path": "/api/predict", "method": "GET", "params": ["symbol", "field", "date"], "description": "Dự đoán giá cổ phiếu đến ngày cụ thể"},
            {"path": "/api/custom-data", "method": "POST", "body": ["file"], "description": "Xử lý dữ liệu từ file tải lên"},
            {"path": "/api/predict-custom", "method": "POST", "body": ["file", "field", "date"], "description": "Dự đoán với dữ liệu tùy chỉnh"}
        ]
    })

@app.route('/api/stocks', methods=['GET'])
def get_stocks():
    stocks = ["ACB", "BID", "CMC", "CMG", "CTG", "DXG", "FPT", "PVC", 
              "MSN", "PNJ", "PDR", "ELC", "KDH", "GAS", "VCB", "VIC", "VNM"]
    return jsonify({"stocks": stocks})

@app.route('/api/stock-data', methods=['GET'])
def get_stock_data():
    symbol = request.args.get('symbol')
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    
    if not symbol or not start_date or not end_date:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
    
    stock_data = get_stock_data_vndirect(symbol, start_date, end_date)
    
    if stock_data is None:
        return jsonify({"error": "No data available for the selected stock and time period"}), 404
    
    stock_data_json = stock_data.reset_index().to_dict(orient='records')
    return jsonify({
        "data": stock_data_json,
        "fields": stock_data.columns.tolist()
    })

@app.route('/api/price-movement', methods=['GET'])
def get_price_movement():
    symbol = request.args.get('symbol')
    field = request.args.get('field')
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    
    if not symbol or not field or not start_date or not end_date:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
    
    stock_data = get_stock_data_vndirect(symbol, start_date, end_date)
    
    if stock_data is None or field not in stock_data.columns:
        return jsonify({"error": "No data available for the selected stock and field"}), 404
    
    chart_data = create_price_movement_chart(stock_data, field, symbol)
    return jsonify(chart_data)

@app.route('/api/avg-profit', methods=['GET'])
def get_avg_profit():
    symbol = request.args.get('symbol')
    field = request.args.get('field')
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    
    if not symbol or not field or not start_date or not end_date:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
    
    stock_data = get_stock_data_vndirect(symbol, start_date, end_date)
    
    if stock_data is None or field not in stock_data.columns:
        return jsonify({"error": "No data available for the selected stock and field"}), 404
    
    r_t = np.log((stock_data[field]/stock_data[field].shift(1)))
    r_t = r_t.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(r_t) == 0:
        return jsonify({"error": "Không thể tính toán tỷ lệ lợi nhuận"}), 400
        
    mean = np.mean(r_t)
    
    chart_data = create_avg_profit_chart(r_t, mean, symbol)
    return jsonify(chart_data)

@app.route('/api/predict', methods=['GET'])
def get_prediction():
    symbol = request.args.get('symbol')
    field = request.args.get('field')
    prediction_date = request.args.get('date')
    
    if not symbol or not field or not prediction_date:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        prediction_date = datetime.datetime.strptime(prediction_date, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
    
    today = datetime.datetime.today().date()
    d30bf = today - BDay(100)
    
    stock_data = get_stock_data_vndirect(symbol, d30bf, today)
    
    if stock_data is None:
        return jsonify({"error": "No data available for the selected stock"}), 404
    
    if field not in stock_data.columns:
        return jsonify({"error": f"Field '{field}' not found in the stock data"}), 400
    
    prediction_results = predict(stock_data, field, symbol, prediction_date)
    
    if "error" in prediction_results:
        return jsonify(prediction_results), 400
        
    return jsonify(prediction_results)

@app.route('/api/custom-data', methods=['POST'])
def process_custom_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        df = pd.read_csv(file, index_col=0, parse_dates=True, infer_datetime_format=True)
        
        if df.empty:
            return jsonify({"error": "Uploaded file contains no data"}), 400
            
        return jsonify({
            "data": df.reset_index().to_dict(orient='records'),
            "fields": df.columns.tolist()
        })
    except Exception as e:
        logger.error(f"Lỗi khi xử lý file tải lên: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/api/predict-custom', methods=['POST'])
def predict_custom():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    field = request.form.get('field')
    prediction_date = request.form.get('date')
    
    if file.filename == '' or not field or not prediction_date:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        prediction_date = datetime.datetime.strptime(prediction_date, '%Y-%m-%d').date()
        df = pd.read_csv(file, index_col=0, parse_dates=True, infer_datetime_format=True)
        
        if df.empty:
            return jsonify({"error": "Uploaded file contains no data"}), 400
            
        if field not in df.columns:
            return jsonify({"error": f"Field '{field}' not found in the uploaded data"}), 400
        
        df.reset_index(inplace=True)
        
        prediction_results = predict(df, field, 'Your_stock', prediction_date)
        
        if "error" in prediction_results:
            return jsonify(prediction_results), 400
            
        return jsonify(prediction_results)
    except Exception as e:
        logger.error(f"Lỗi khi dự đoán với dữ liệu tùy chỉnh: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    if not os.path.exists('Model'):
        os.makedirs('Model')
    
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    app.run(debug=debug_mode, port=port, host='0.0.0.0')