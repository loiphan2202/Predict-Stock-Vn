import pandas as pd
from vnstock import stock_historical_data

# Danh sách các ngành và mã cổ phiếu
industries = {
    "Ngân hàng": ["VCB", "BID", "CTG", "ACB"],
    "Bất động sản": ["VIC", "KDH", "PDR", "DXG"],
    "Công nghệ": ["FPT", "CMG", "CMC", "ELC"],
    "Bán lẻ": ["PNJ", "MSN", "VNM"],
    "Năng lượng": ["GAS", "PVC"]
}

# Thiết lập ngày lấy dữ liệu
start_date = '2014-04-01'
end_date = '2024-04-01'

# Lấy dữ liệu cho từng mã cổ phiếu
for sector, stocks in industries.items():
    for stock in stocks:
        try:
            print(f"Fetching data for {stock}...")
            df = stock_historical_data(symbol=stock, start_date=start_date, end_date=end_date, resolution='1D')

            
            df.rename(columns={
                'time': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)

            # Xóa cột 'ticker'
            if 'ticker' in df.columns:
                df.drop(columns=['ticker'], inplace=True)

            
            # Sắp xếp lại thứ tự cột để Adj Close đứng trước Volume
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]


            # Thêm cột Adj Close (giữ nguyên giá Close)
            df['Adj Close'] = df['Close']

            # Lưu vào file CSV riêng
            file_name = f"data_{stock}.csv"
            df.to_csv(file_name, index=False)
            print(f"✅ Saved {file_name}")

        except Exception as e:
            print(f"❌ Error fetching data for {stock}: {e}")
