!pip install pandas
import numpy as np
# hỗ trợ cho việc tính toán các mảng nhiều chiều
import pandas as pd
# thao tác và phân tích dữ liệu

import matplotlib.pyplot as plt
# vẽ biểu đồ, đồ thị
%matplotlib inline
from sklearn.preprocessing import MinMaxScaler
# hàm đưa dữ liệu về giá trị (0,1)
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
# Gộp dữ liệu vào chuyển thành dữ liệu time series
from keras.models import Sequential
# khởi tạo mạng neurol
from keras.layers import Dense
# một lớp để chuyển dữ liệu từ lớp input vào model ????
from keras.layers import LSTM
# mô hình LSTM
from keras.layers import Dropout
# giúp bỏ bớt các node, giúp lọc lại những node có thông tin cần thiết
from tensorflow import keras
# from datetime import datatime
plt.style.use("fivethirtyeight")
# style của thư viện mathplotlib
import warnings
warnings.filterwarnings("ignore")
# Những thư viện cần dùng
from keras.losses import MeanSquaredError

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import seaborn as sns

from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

plt.style.use("fivethirtyeight")
import warnings
warnings.filterwarnings("ignore")

from google.colab import drive
drive.mount('/content/drive')
!ls /content/drive/MyDrive/Hocmay

# Danh sách mã cổ phiếu đang traintrain
available_tickers = ["ACB", "BID", "CMC", "CMG", "CTG", "DXG", "FPT", "PVC",
                     "MSN", "PNJ", "PDR", "ELC", "KDH", "GAS", "VCB", "VIC", "VNM"]

Macophieu = "VNM"

prop = "Close"  #"Open", "High", "Low", "Close", "Volume"

folder_path = "/content/drive/MyDrive/Hocmay"

file_path = f"{folder_path}/{Macophieu}.csv"

try:
    # Đọc dữ liệu từ CSV
    data = pd.read_csv(file_path)

    if prop not in data.columns:
        raise ValueError(f"Cột '{prop}' không tồn tại trong dữ liệu.")

    # Chuyển đổi cột 'Date' sang dạng datetime
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values(by='Date')

    # Lọc dữ liệu chỉ lấy cột Date và prop
    selected_data = data[['Date', prop]]

    # 5 dòng đầu tiên của dữ liệu
    print(selected_data.head())

except FileNotFoundError:
    print(f"{Macophieu}.csv không tồn tại trong thư mục {folder_path}.")
except ValueError as e:
    print(f"Lỗi: {e}")
except Exception as e:
    print(f"Lỗi không xác định: {e}")

plt.figure(figsize = (16,6))
plt.plot(data['Date'], data[prop])
plt.title(f"{Macophieu} - {prop} history")
plt.xlabel('Date', fontsize = 18)
plt.ylabel(f'{prop}', fontsize = 18)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


print(f"\nThống kê mô tả của {prop}:")
print(data[prop].describe())


plt.figure(figsize=(12, 6))
sns.histplot(data[prop], kde=True)
plt.title(f"Phân phối giá {prop} của {Macophieu}")
plt.tight_layout()
plt.show()

# 1. Thêm hàm tạo nhãn tăng/giảm và thêm cột nhãn vào DataFrame

def label_change(pct):
    if pct > 0.03:
        return 2  # tăng mạnh
    elif pct > 0.01:
        return 1  # tăng nhẹ
    elif pct > -0.01:
        return 0  # không đổi/ít thay đổi
    elif pct > -0.03:
        return -1 # giảm nhẹ
    else:
        return -2 # giảm mạnh

# Thêm cột nhãn sau khi đọc dữ liệu và trước khi chia train/test:
data['pct_change'] = data[prop].pct_change()
data['label'] = data['pct_change'].apply(label_change)

# Sau khi tạo cột label, thêm cột final_price vào DataFrame
alpha = 0.01  # hệ số điều chỉnh
# final_price = giá thực + nhãn * alpha * giá thực
# Chỉ tính cho tập train/test, không tính cho giá đầu tiên (do pct_change)
data['final_price'] = data[prop] + data['label'] * alpha * data[prop]

data_end = int(np.floor(0.8*(data.shape[0])))
# lấy cái mốc là data_end theo tử lệ 2:8
train = data[0:data_end][prop]
train_final = data[0:data_end]['final_price']
test = data[data_end:][prop]
test_final = data[data_end:]['final_price']
date_test = data[data_end:]['Date']
# lấy 20% data là giá mở cho tập test

# Hiển thị phân chia dữ liệu
plt.figure(figsize=(16, 6))
plt.plot(data['Date'][:data_end], train, 'b-', label='Training Data')
plt.plot(data['Date'][data_end:], test, 'r-', label='Testing Data')
plt.title(f"{Macophieu} - Training/Testing Split")
plt.xlabel('Date')
plt.ylabel(f'{prop}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Số lượng dữ liệu train: {len(train)}")
print(f"Số lượng dữ liệu test: {len(test)}")

train = train.values.reshape(-1)
test = test.values.reshape(-1)
date_test = date_test.values.reshape(-1)


def get_data(train, test, time_step, num_predict, date, train_final=None, test_final=None):
    x_train = []
    y_train = []
    y_train_final = []
    x_test = []
    y_test = []
    y_test_final = []
    date_test = []
    for i in range(0, len(train) - time_step - num_predict):
        x_train.append(train[i:i+time_step])
        y_train.append(train[i+time_step:i+time_step+num_predict])
        if train_final is not None:
            y_train_final.append(train_final[i+time_step:i+time_step+num_predict])
    for i in range(0, len(test) - time_step - num_predict):
        x_test.append(test[i:i+time_step])
        y_test.append(test[i+time_step:i+time_step+num_predict])
        if test_final is not None:
            y_test_final.append(test_final[i+time_step:i+time_step+num_predict])
        if i + time_step + num_predict <= len(date):
            date_test.append(date[i+time_step:i+time_step+num_predict][0])
    return (np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test),
            np.asarray(date_test), np.asarray(y_train_final), np.asarray(y_test_final))

x_train, y_train, x_test, y_test, date_test, y_train_final, y_test_final = get_data(
    train, test, 30, 1, date_test, train_final, test_final)

# chuyển về dạng ma trận đưa vào minmaxscaler()
x_train = x_train.reshape(-1,30)
x_test = x_test.reshape(-1,30)

# dua ve 0->1 cho tap train
scaler = MinMaxScaler()

# gọi hàm scaler để nén hoặc giải nén data về khoảng (0,1) để máy hiểu góp phần tăng tốc độ máy học
# fit_transform nén data lại cho model cho 4 ma trận x, y_train x,y test
x_train = scaler.fit_transform(x_train)
y_train = scaler.fit_transform(y_train)

x_test = scaler.fit_transform(x_test)
y_test = scaler.fit_transform(y_test)

# Thêm scaler cho final_price
y_train_final = scaler.fit_transform(y_train_final)
y_test_final = scaler.fit_transform(y_test_final)

# chuyển về dạng ma trận đưa vào keras() thêm một chiều thứ 3 để có bias => để thành ma trận 3D cho phù hợp với bài toán

# Reshape lai cho x_train
x_train = x_train.reshape(-1,30,1)
y_train = y_train.reshape(-1,1)

#reshape lai cho test
x_test = x_test.reshape(-1,30,1)
y_test = y_test.reshape(-1,1)
date_test = date_test.reshape(-1,1)

# Reshape lai cho y_train_final, y_test_final
y_train_final = y_train_final.reshape(-1, 1)
y_test_final = y_test_final.reshape(-1, 1)

n_input = 30
n_features = 1

from keras.layers import Input
from keras.models import Model

input_layer = Input(shape=(n_input, n_features))
x = LSTM(50, return_sequences=True)(input_layer)
x = Dropout(0.3)(x)
x = LSTM(50, return_sequences=True)(x)
x = Dropout(0.3)(x)
x = LSTM(50)(x)
x = Dropout(0.3)(x)

output_final_price = Dense(1, name='final_price')(x)
model = Model(inputs=input_layer, outputs=output_final_price)
model.compile(optimizer='adam', loss='mse')
model.summary()

%cd /content/drive/MyDrive/Hocmay
# fit đưa tất cả vào model
model.fit(x_train, y_train_final, epochs=200, validation_split=0.2, verbose=1, batch_size=30)
model.save(f'{Macophieu}_{prop}_finalprice.h5')

try:
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 1, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
except NameError:
    print("Error: The 'history' variable is not defined. Make sure the model training completed successfully.")
except Exception as e:
    print(f"Error plotting training history: {e}")


%cd /content/drive/MyDrive/Hocmay
# Test model vừa train
model = keras.models.load_model(f'{Macophieu}_{prop}_finalprice.h5', custom_objects={"mse": MeanSquaredError()})
# load model

final_price_pred = model.predict(x_test)
final_price_pred = scaler.inverse_transform(final_price_pred)
final_price_actual = scaler.inverse_transform(y_test_final)

# Chuyển đổi date_test thành datetime để vẽ đồ thị tốt hơn
date_test_pd = pd.to_datetime(date_test.flatten())

# Tạo DataFrame cho kết quả
results_df = pd.DataFrame({
    'Date': date_test_pd,
    'Actual': final_price_actual.flatten(),
    'Predicted': final_price_pred.flatten()
})

# Tính toán lỗi dự đoán
results_df['Error'] = results_df['Actual'] - results_df['Predicted']
results_df['Absolute Error'] = np.abs(results_df['Error'])
results_df['Squared Error'] = results_df['Error'] ** 2
results_df['Percent Error'] = np.abs(results_df['Error'] / results_df['Actual']) * 100


# Hiển thị bảng kết quả
print("\nKết quả dự đoán (5 dòng đầu tiên):")
print(results_df.head().to_string())

# Tính toán số liệu
mae = mean_absolute_error(final_price_actual, final_price_pred)
rmse = np.sqrt(mean_squared_error(final_price_actual, final_price_pred))
mse = mean_squared_error(final_price_actual, final_price_pred)
r2 = r2_score(final_price_actual, final_price_pred)
mape = np.mean(np.abs((final_price_actual - final_price_pred) / final_price_actual)) * 100

# Đối với số liệu phân loại, sẽ coi một dự đoán là đúng nếu nó dự đoán đúng hướng
actual_direction = np.sign(np.diff(final_price_actual.flatten(), prepend=final_price_actual[0]))
predicted_direction = np.sign(np.diff(final_price_pred.flatten(), prepend=final_price_pred[0]))

# Tính toán độ chính xác của dự đoán hướng
direction_accuracy = np.mean(actual_direction == predicted_direction)

# Tính toán TP, FP, TN, FN để dự đoán tăng/giảm
# Tích cực = Lên, Tiêu cực = Xuống
tp = np.sum((actual_direction == 1) & (predicted_direction == 1))
fp = np.sum((actual_direction != 1) & (predicted_direction == 1))
tn = np.sum((actual_direction != 1) & (predicted_direction != 1))
fn = np.sum((actual_direction == 1) & (predicted_direction != 1))

# Tính độ chính xác, độ thu hồi, f1
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Hiển thị số liệu theo định dạng bảng
metrics_data = {
    'Metric': ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE (%)', 'Direction Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [mae, mse, rmse, r2, mape, direction_accuracy, precision, recall, f1]
}
metrics_df = pd.DataFrame(metrics_data)
print("\nCác chỉ số đánh giá mô hình:")
print(metrics_df.to_string(index=False))

# Tạo ma trận nhầm lẫn để dự đoán hướng
cm = np.array([[tn, fp], [fn, tp]])
labels = ['Down', 'Up']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (Direction Prediction)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# Biểu đồ giá thực tế so với giá dự kiến
plt.figure(figsize=(16, 8))
plt.subplot(2, 1, 1)
plt.plot(date_test_pd[30:], final_price_actual[30:], 'b-', label='Actual Price')
plt.plot(date_test_pd[30:], final_price_pred[30:], 'r-', label='Predicted Price')
plt.title(f"{Macophieu} - {prop} Price Prediction")
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)

# Biểu đồ dự đoán sai
plt.subplot(2, 1, 2)
plt.plot(date_test_pd[30:], results_df['Error'][30:], 'g-')
plt.fill_between(date_test_pd[30:], results_df['Error'][30:], 0, alpha=0.3, color='g')
plt.title('Prediction Error')
plt.ylabel('Error')
plt.xlabel('Date')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Phân phối lỗi đồ thị
plt.figure(figsize=(12, 6))
sns.histplot(results_df['Error'], kde=True)
plt.title('Distribution of Prediction Errors')
plt.xlabel('Error')
plt.axvline(x=0, color='r', linestyle='--')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Tạo biểu đồ phân tán của giá trị thực tế so với giá trị dự đoán
plt.figure(figsize=(10, 10))
plt.scatter(final_price_actual, final_price_pred, alpha=0.5)
plt.plot([final_price_actual.min(), final_price_actual.max()], [final_price_actual.min(), final_price_actual.max()], 'r--')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()

# Dự báo giá trị tiếp theo
print("\nDự đoán giá tiếp theo:")
print(f'Giá dự đoán của ngày hôm qua: {final_price_pred[-1][0]:.2f}')

# Hiển thị báo cáo tóm tắt
print("\n" + "="*50)
print(f"TÓM TẮT MÔ HÌNH DỰ ĐOÁN GIÁ {prop} CHO MÃ {Macophieu}")
print("="*50)
print(f"Số lượng dữ liệu train: {len(train)}")
print(f"Số lượng dữ liệu test: {len(test)}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Độ chính xác dự đoán xu hướng: {direction_accuracy*100:.2f}%")
print(f"F1 Score cho dự đoán xu hướng: {f1:.4f}")
print("="*50)

# ============== Phân tích ==============

# 1. Phân tích kết quả xử lý dữ liệu
print("\n=== Data Processing Results ===")
print("\n1.1 Data Overview:")
print(f"Total number of data points: {len(data)}")
print(f"Date range: from {data['Date'].min()} to {data['Date'].max()}")
print(f"Number of trading days: {len(data['Date'].unique())}")

# Tính toán số liệu thống kê cơ bản cho tất cả các cột giá
price_columns = ['Open', 'High', 'Low', 'Close']
print("\n1.2 Statistical Summary of Price Data:")
stats_df = data[price_columns].describe()
print(stats_df)

# Tính toán lợi nhuận và biến động hàng ngày
data['Daily_Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily_Return'].rolling(window=20).std()

# 2. Hình ảnh trực quan vip
print("\n=== Generating Advanced Visualizations ===")

# 2.1 Xu hướng giá và biến động
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(data['Date'], data['Close'], label='Close Price')
plt.title(f'{Macophieu} Stock Price and Volatility Analysis')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(data['Date'], data['Volatility'], color='red', label='20-day Volatility')
plt.title('Price Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2.2 Phân tích tương quan
plt.figure(figsize=(10, 8))
correlation_matrix = data[price_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Price Correlation Matrix')
plt.tight_layout()
plt.show()

# 2.3 Phân tích hiệu suất mô hình
print("\n=== Model Performance Analysis ===")

# Tạo bảng số liệu hiệu suất toàn diện
performance_metrics = {
    'Metric': ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE (%)', 'Direction Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [mae, mse, rmse, r2, mape, direction_accuracy, precision, recall, f1]
}
performance_df = pd.DataFrame(performance_metrics)
print("\n2.1 Model Performance Metrics:")
print(performance_df.to_string(index=False))

# 2.4 Phân tích lỗi dự đoán
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(date_test_pd, results_df['Error'], label='Prediction Error')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Prediction Error Over Time')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
sns.histplot(results_df['Error'], kde=True)
plt.title('Distribution of Prediction Errors')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.axvline(x=0, color='r', linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Thêm hàm dự đoán cuốn chiếu (multi-step prediction)
def recursive_predict(model, last_30_days, scaler, n_steps=1):
    """
    Dự đoán liên tiếp n_steps ngày tương lai dựa trên 30 giá trị gần nhất (có thể là giá thực hoặc giá dự đoán trước đó).
    Trả về list giá dự đoán và nhãn tăng/giảm tương ứng.
    """
    input_seq = last_30_days.copy()
    preds = []
    labels = []
    for _ in range(n_steps):
        scaled_input = scaler.transform(np.array(input_seq).reshape(1, -1))
        scaled_input = scaled_input.reshape((1, 30, 1))
        pred_scaled = model.predict(scaled_input, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0, 0]
        preds.append(pred)
        # Tính nhãn tăng/giảm so với giá cuối cùng của input_seq
        pct = (pred - input_seq[-1]) / input_seq[-1]
        labels.append(label_change(pct))
        # Cập nhật input_seq cho bước tiếp theo
        input_seq = input_seq[1:] + [pred]
    return preds, labels

# Ví dụ sử dụng (bạn có thể chạy ở cuối notebook):
# last_30 = list(data[prop].values)[-30:]
# preds, labels = recursive_predict(model, last_30, scaler, n_steps=7)
# print('Giá dự đoán:', preds)
# print('Nhãn tăng/giảm:', labels)

# y_train: (số mẫu, 1) giá
# y_label: (số mẫu, 1) nhãn số
y_train_price = y_train  # đã có
y_train_label = ...      # tạo từ nhãn tăng/giảm tương ứng với từng sample (cần đồng bộ index với y_train)

# Chia train/test cho cả giá và nhãn
train = data[0:data_end][prop]
train_label = data[0:data_end]['label']
train_final = data[0:data_end]['final_price']
test = data[data_end:][prop]
test_label = data[data_end:]['label']
test_final = data[data_end:]['final_price']
date_test = data[data_end:]['Date']

# Chuyển về numpy
train = train.values.reshape(-1)
test = test.values.reshape(-1)
train_label = train_label.values.reshape(-1)
test_label = test_label.values.reshape(-1)
date_test = date_test.values.reshape(-1)

# Hàm tạo dữ liệu với 2 feature: giá và nhãn

def get_data_with_label(train, train_label, test, test_label, time_step, num_predict, date, train_final=None, test_final=None):
    x_train = []
    y_train = []
    y_train_final = []
    x_test = []
    y_test = []
    y_test_final = []
    date_test_out = []
    for i in range(0, len(train) - time_step - num_predict):
        x_feat = np.stack([train[i:i+time_step], train_label[i:i+time_step]], axis=1)
        x_train.append(x_feat)
        y_train.append(train[i+time_step:i+time_step+num_predict])
        if train_final is not None:
            y_train_final.append(train_final[i+time_step:i+time_step+num_predict])
    for i in range(0, len(test) - time_step - num_predict):
        x_feat = np.stack([test[i:i+time_step], test_label[i:i+time_step]], axis=1)
        x_test.append(x_feat)
        y_test.append(test[i+time_step:i+time_step+num_predict])
        if test_final is not None:
            y_test_final.append(test_final[i+time_step:i+time_step+num_predict])
        if i + time_step + num_predict <= len(date):
            date_test_out.append(date[i+time_step:i+time_step+num_predict][0])
    return (np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test),
            np.asarray(date_test_out), np.asarray(y_train_final), np.asarray(y_test_final))

x_train, y_train, x_test, y_test, date_test, y_train_final, y_test_final = get_data_with_label(
    train, train_label, test, test_label, 30, 1, date_test, train_final.values, test_final.values)

# Dua ve 0-1 cho tap train/test (chỉ scale giá và final_price, không scale nhãn)
scaler = MinMaxScaler()
scaler_final = MinMaxScaler()

# Scale giá trong input
x_train_price = scaler.fit_transform(x_train[:,:,0].reshape(-1, 1)).reshape(x_train.shape[0], x_train.shape[1])
x_test_price = scaler.transform(x_test[:,:,0].reshape(-1, 1)).reshape(x_test.shape[0], x_test.shape[1])
# Nhãn giữ nguyên
x_train_label = x_train[:,:,1]
x_test_label = x_test[:,:,1]
# Gộp lại
x_train_scaled = np.stack([x_train_price, x_train_label], axis=2)
x_test_scaled = np.stack([x_test_price, x_test_label], axis=2)

# Scale y_train, y_test, y_train_final, y_test_final (reshape về (-1,1) trước khi scale)
y_train = scaler.transform(y_train.reshape(-1, 1))
y_test = scaler.transform(y_test.reshape(-1, 1))
y_train_final = scaler_final.fit_transform(y_train_final.reshape(-1, 1))
y_test_final = scaler_final.transform(y_test_final.reshape(-1, 1))

# Reshape lại cho phù hợp
x_train = x_train_scaled
x_test = x_test_scaled
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train_final = y_train_final.reshape(-1, 1)
y_test_final = y_test_final.reshape(-1, 1)
date_test = date_test.reshape(-1, 1)

n_input = 30
n_features = 2

from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Model
input_layer = Input(shape=(n_input, n_features))
x = LSTM(50, return_sequences=True)(input_layer)
x = Dropout(0.3)(x)
x = LSTM(50, return_sequences=True)(x)
x = Dropout(0.3)(x)
x = LSTM(50)(x)
x = Dropout(0.3)(x)
output_final_price = Dense(1, name='final_price')(x)
output_label = Dense(1, activation='linear', name='label')(x)
model = Model(inputs=input_layer, outputs=[output_final_price, output_label])
model.compile(optimizer='adam', loss={'final_price': 'mse', 'label': 'mse'}, loss_weights={'final_price': 1.0, 'label': 0.5})
model.summary()

%cd /content/drive/MyDrive/Hocmay
# Lấy nhãn cho tập train/test (dự đoán nhãn của ngày tiếp theo)
y_train_label = []
y_test_label = []
for i in range(0, len(train_label) - n_input - 1):
    y_train_label.append(train_label[i+n_input+1-1])
for i in range(0, len(test_label) - n_input - 1):
    y_test_label.append(test_label[i+n_input+1-1])
y_train_label = np.array(y_train_label).reshape(-1, 1)
y_test_label = np.array(y_test_label).reshape(-1, 1)

model.fit(x_train, {'final_price': y_train_final, 'label': y_train_label}, epochs=200, validation_split=0.2, verbose=1, batch_size=30)
model.save(f'{Macophieu}_{prop}_finalprice_labelinput_multiout.h5')

# Test model vừa train
model = keras.models.load_model(f'{Macophieu}_{prop}_finalprice_labelinput_multiout.h5', custom_objects={"mse": MeanSquaredError()})
final_price_pred, label_pred = model.predict(x_test)
final_price_pred = scaler_final.inverse_transform(final_price_pred)
final_price_actual = scaler_final.inverse_transform(y_test_final)

# Đánh giá xu hướng dựa trên label_pred (so sánh với y_test_label)
label_pred_rounded = np.round(label_pred).astype(int)
label_actual = y_test_label.astype(int)
direction_accuracy = np.mean(label_pred_rounded.flatten() == label_actual.flatten())

# Tính toán lỗi dự đoán
results_df['Error'] = results_df['Actual'] - results_df['Predicted']
results_df['Absolute Error'] = np.abs(results_df['Error'])
results_df['Squared Error'] = results_df['Error'] ** 2
results_df['Percent Error'] = np.abs(results_df['Error'] / results_df['Actual']) * 100


# Hiển thị bảng kết quả
print("\nKết quả dự đoán (5 dòng đầu tiên):")
print(results_df.head().to_string())

# Tính toán số liệu
mae = mean_absolute_error(final_price_actual, final_price_pred)
rmse = np.sqrt(mean_squared_error(final_price_actual, final_price_pred))
mse = mean_squared_error(final_price_actual, final_price_pred)
r2 = r2_score(final_price_actual, final_price_pred)
mape = np.mean(np.abs((final_price_actual - final_price_pred) / final_price_actual)) * 100

# Đối với số liệu phân loại, sẽ coi một dự đoán là đúng nếu nó dự đoán đúng hướng
actual_direction = np.sign(np.diff(final_price_actual.flatten(), prepend=final_price_actual[0]))
predicted_direction = np.sign(np.diff(final_price_pred.flatten(), prepend=final_price_pred[0]))

# Tính toán độ chính xác của dự đoán hướng
direction_accuracy = np.mean(actual_direction == predicted_direction)

# Tính toán TP, FP, TN, FN để dự đoán tăng/giảm
# Tích cực = Lên, Tiêu cực = Xuống
tp = np.sum((actual_direction == 1) & (predicted_direction == 1))
fp = np.sum((actual_direction != 1) & (predicted_direction == 1))
tn = np.sum((actual_direction != 1) & (predicted_direction != 1))
fn = np.sum((actual_direction == 1) & (predicted_direction != 1))

# Tính độ chính xác, độ thu hồi, f1
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Hiển thị số liệu theo định dạng bảng
metrics_data = {
    'Metric': ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE (%)', 'Direction Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [mae, mse, rmse, r2, mape, direction_accuracy, precision, recall, f1]
}
metrics_df = pd.DataFrame(metrics_data)
print("\nCác chỉ số đánh giá mô hình:")
print(metrics_df.to_string(index=False))

# Tạo ma trận nhầm lẫn để dự đoán hướng
cm = np.array([[tn, fp], [fn, tp]])
labels = ['Down', 'Up']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (Direction Prediction)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# Biểu đồ giá thực tế so với giá dự kiến
plt.figure(figsize=(16, 8))
plt.subplot(2, 1, 1)
plt.plot(date_test_pd[30:], final_price_actual[30:], 'b-', label='Actual Price')
plt.plot(date_test_pd[30:], final_price_pred[30:], 'r-', label='Predicted Price')
plt.title(f"{Macophieu} - {prop} Price Prediction")
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)

# Biểu đồ dự đoán sai
plt.subplot(2, 1, 2)
plt.plot(date_test_pd[30:], results_df['Error'][30:], 'g-')
plt.fill_between(date_test_pd[30:], results_df['Error'][30:], 0, alpha=0.3, color='g')
plt.title('Prediction Error')
plt.ylabel('Error')
plt.xlabel('Date')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Phân phối lỗi đồ thị
plt.figure(figsize=(12, 6))
sns.histplot(results_df['Error'], kde=True)
plt.title('Distribution of Prediction Errors')
plt.xlabel('Error')
plt.axvline(x=0, color='r', linestyle='--')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Tạo biểu đồ phân tán của giá trị thực tế so với giá trị dự đoán
plt.figure(figsize=(10, 10))
plt.scatter(final_price_actual, final_price_pred, alpha=0.5)
plt.plot([final_price_actual.min(), final_price_actual.max()], [final_price_actual.min(), final_price_actual.max()], 'r--')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()

# Dự báo giá trị tiếp theo
print("\nDự đoán giá tiếp theo:")
print(f'Giá dự đoán của ngày hôm qua: {final_price_pred[-1][0]:.2f}')

# Hiển thị báo cáo tóm tắt
print("\n" + "="*50)
print(f"TÓM TẮT MÔ HÌNH DỰ ĐOÁN GIÁ {prop} CHO MÃ {Macophieu}")
print("="*50)
print(f"Số lượng dữ liệu train: {len(train)}")
print(f"Số lượng dữ liệu test: {len(test)}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Độ chính xác dự đoán xu hướng: {direction_accuracy*100:.2f}%")
print(f"F1 Score cho dự đoán xu hướng: {f1:.4f}")
print("="*50)

# ============== Phân tích ==============

# 1. Phân tích kết quả xử lý dữ liệu
print("\n=== Data Processing Results ===")
print("\n1.1 Data Overview:")
print(f"Total number of data points: {len(data)}")
print(f"Date range: from {data['Date'].min()} to {data['Date'].max()}")
print(f"Number of trading days: {len(data['Date'].unique())}")

# Tính toán số liệu thống kê cơ bản cho tất cả các cột giá
price_columns = ['Open', 'High', 'Low', 'Close']
print("\n1.2 Statistical Summary of Price Data:")
stats_df = data[price_columns].describe()
print(stats_df)

# Tính toán lợi nhuận và biến động hàng ngày
data['Daily_Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily_Return'].rolling(window=20).std()

# 2. Hình ảnh trực quan vip
print("\n=== Generating Advanced Visualizations ===")

# 2.1 Xu hướng giá và biến động
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(data['Date'], data['Close'], label='Close Price')
plt.title(f'{Macophieu} Stock Price and Volatility Analysis')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(data['Date'], data['Volatility'], color='red', label='20-day Volatility')
plt.title('Price Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2.2 Phân tích tương quan
plt.figure(figsize=(10, 8))
correlation_matrix = data[price_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Price Correlation Matrix')
plt.tight_layout()
plt.show()

# 2.3 Phân tích hiệu suất mô hình
print("\n=== Model Performance Analysis ===")

# Tạo bảng số liệu hiệu suất toàn diện
performance_metrics = {
    'Metric': ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE (%)', 'Direction Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [mae, mse, rmse, r2, mape, direction_accuracy, precision, recall, f1]
}
performance_df = pd.DataFrame(performance_metrics)
print("\n2.1 Model Performance Metrics:")
print(performance_df.to_string(index=False))

# 2.4 Phân tích lỗi dự đoán
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(date_test_pd, results_df['Error'], label='Prediction Error')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Prediction Error Over Time')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
sns.histplot(results_df['Error'], kde=True)
plt.title('Distribution of Prediction Errors')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.axvline(x=0, color='r', linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Thêm hàm dự đoán cuốn chiếu (multi-step prediction)
def recursive_predict(model, last_30_days, scaler, n_steps=1):
    """
    Dự đoán liên tiếp n_steps ngày tương lai dựa trên 30 giá trị gần nhất (có thể là giá thực hoặc giá dự đoán trước đó).
    Trả về list giá dự đoán và nhãn tăng/giảm tương ứng.
    """
    input_seq = last_30_days.copy()
    preds = []
    labels = []
    for _ in range(n_steps):
        scaled_input = scaler.transform(np.array(input_seq).reshape(1, -1))
        scaled_input = scaled_input.reshape((1, 30, 1))
        pred_scaled = model.predict(scaled_input, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0, 0]
        preds.append(pred)
        # Tính nhãn tăng/giảm so với giá cuối cùng của input_seq
        pct = (pred - input_seq[-1]) / input_seq[-1]
        labels.append(label_change(pct))
        # Cập nhật input_seq cho bước tiếp theo
        input_seq = input_seq[1:] + [pred]
    return preds, labels

# Ví dụ sử dụng (bạn có thể chạy ở cuối notebook):
# last_30 = list(data[prop].values)[-30:]
# preds, labels = recursive_predict(model, last_30, scaler, n_steps=7)
# print('Giá dự đoán:', preds)
# print('Nhãn tăng/giảm:', labels)

# y_train: (số mẫu, 1) giá
# y_label: (số mẫu, 1) nhãn số
y_train_price = y_train  # đã có
y_train_label = ...      # tạo từ nhãn tăng/giảm tương ứng với từng sample (cần đồng bộ index với y_train)

