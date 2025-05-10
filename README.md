<!-- <<<<<<< HEAD
# Stock Price Prediction API

API dự đoán giá cổ phiếu sử dụng thuật toán LSTM. API này cung cấp các endpoint để lấy dữ liệu cổ phiếu, tạo biểu đồ và dự đoán giá trong tương lai.

## Cài đặt

### Yêu cầu
- Python 3.8 trở lên
- Các thư viện được liệt kê trong file requirements.txt

### Bước cài đặt

1. Clone hoặc giải nén mã nguồn vào thư mục của bạn
2. Cài đặt các thư viện cần thiết:
```
pip install -r requirements.txt
```
3. Khởi động API:
```
python api.py
```

API sẽ chạy mặc định tại http://localhost:5000

## Cấu hình

Bạn có thể thiết lập các biến môi trường để cấu hình API:
- `FLASK_DEBUG`: Đặt thành "true" để bật chế độ debug
- `FLASK_PORT`: Thay đổi cổng chạy API (mặc định: 5000)

Ví dụ:
```
FLASK_DEBUG=true FLASK_PORT=8000 python api.py
```

## Các endpoint chính

### GET `/api/stocks`
Lấy danh sách các mã cổ phiếu có sẵn

### GET `/api/stock-data`
Lấy dữ liệu cổ phiếu trong khoảng thời gian

Tham số:
- `symbol`: Mã cổ phiếu (ví dụ: FPT)
- `start`: Ngày bắt đầu (định dạng: YYYY-MM-DD)
- `end`: Ngày kết thúc (định dạng: YYYY-MM-DD)

### GET `/api/predict`
Dự đoán giá cổ phiếu đến ngày cụ thể

Tham số:
- `symbol`: Mã cổ phiếu (ví dụ: FPT)
- `field`: Trường giá cần dự đoán (open, high, low, close)
- `date`: Ngày dự đoán (định dạng: YYYY-MM-DD)

### POST `/api/custom-data`
Xử lý dữ liệu từ file CSV tải lên

### POST `/api/predict-custom`
Dự đoán với dữ liệu tùy chỉnh

## Ví dụ sử dụng với React

```javascript
// Lấy dữ liệu cổ phiếu
const fetchStockData = async () => {
  const response = await fetch('http://localhost:5000/api/stock-data?symbol=FPT&start=2023-01-01&end=2023-12-31');
  const data = await response.json();
  return data;
}

// Dự đoán giá cổ phiếu
const predictStock = async () => {
  const response = await fetch('http://localhost:5000/api/predict?symbol=FPT&field=close&date=2024-06-30');
  const prediction = await response.json();
  return prediction;
}
```

## Xử lý lỗi khi VNDIRECT bảo trì


## Lưu ý

- API VNDIRECT đang bảo trì đến ngày 05/05/2025
- Mô hình sẽ được tự động khởi tạo nếu không tồn tại trong thư mục Model
- Kiểm tra thêm thông tin về API tại endpoint chính `/` 
=======
# Predict-Stock-Vn
>>>>>>> ff0d7e049fda3ed8afc08a61d96ac058da3c1345 -->
