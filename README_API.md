# Stock Prediction API

This API provides endpoints for retrieving stock data and making price predictions using LSTM models.

## Running the API

1. Install required packages:
```
pip install flask flask-cors requests pandas numpy tensorflow keras sklearn matplotlib seaborn
```

2. Start the API server:
```
python api.py
```

The server will run on http://localhost:5000 by default.

## API Endpoints

### Get Available Stocks

```
GET /api/stocks
```

Returns a list of available stock symbols that can be used for predictions.

Example response:
```json
["ACB", "BID", "CMC", "CMG", "CTG", "DXG", "FPT", "PVC", "MSN", "PNJ", "PDR", "ELC", "KDH", "GAS", "VCB", "VIC", "VNM"]
```

### Get Available Fields

```
GET /api/fields
```

Returns a list of available data fields that can be used for predictions.

Example response:
```json
["open", "high", "low", "close", "volume"]
```

### Get Historical Stock Data

```
GET /api/stock-data?symbol={SYMBOL}&start_date={START_DATE}&end_date={END_DATE}
```

Parameters:
- `symbol`: Stock symbol (e.g., "VNM")
- `start_date`: Start date in YYYY-MM-DD format
- `end_date`: End date in YYYY-MM-DD format

Returns historical stock data for the specified symbol and date range.

Example response:
```json
{
  "dates": ["2023-01-01", "2023-01-02", ...],
  "open": [100.2, 101.5, ...],
  "high": [102.3, 103.1, ...],
  "low": [99.8, 100.2, ...],
  "close": [101.1, 102.4, ...],
  "volume": [1000000, 1200000, ...]
}
```

### Predict Stock Prices

```
POST /api/predict
```

Request body (JSON):
```json
{
  "symbol": "VNM",
  "field": "close",
  "prediction_days": 5
}
```

Parameters:
- `symbol`: Stock symbol
- `field`: Data field to predict ("open", "high", "low", "close", or "volume")
- `prediction_days`: Number of days to predict (1-30)

Returns predicted values for the specified number of days.

Example response:
```json
{
  "symbol": "VNM",
  "field": "close",
  "dates": ["2023-06-01", "2023-06-02", "2023-06-05", "2023-06-06", "2023-06-07"],
  "predicted_values": [105.2, 106.1, 107.3, 106.8, 108.2]
}
```

### Get Stock Statistics

```
GET /api/stats?symbol={SYMBOL}&field={FIELD}&period={PERIOD}
```

Parameters:
- `symbol`: Stock symbol (required)
- `field`: Data field for statistics (default: "close")
- `period`: Number of days to calculate statistics for (default: 365)

Returns statistical information about a stock.

Example response:
```json
{
  "symbol": "VNM",
  "field": "close",
  "mean_return": 0.0012,
  "std_return": 0.0156,
  "min_value": 95.2,
  "max_value": 110.5,
  "current_value": 108.7,
  "period_change": 15.2
}
```

## Using with React

Here's an example of how to call the API from a React application:

```javascript
// Example: Fetching stock data
const fetchStockData = async (symbol, startDate, endDate) => {
  try {
    const response = await fetch(
      `http://localhost:5000/api/stock-data?symbol=${symbol}&start_date=${startDate}&end_date=${endDate}`
    );
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error fetching stock data:", error);
    return null;
  }
};

// Example: Getting predictions
const getPrediction = async (symbol, field, days) => {
  try {
    const response = await fetch("http://localhost:5000/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        symbol: symbol,
        field: field,
        prediction_days: days,
      }),
    });
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error getting prediction:", error);
    return null;
  }
};
``` 