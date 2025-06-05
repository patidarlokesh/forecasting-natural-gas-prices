# Natural Gas Price Forecast using Linear Regression

## 🔍 Overview
This project was developed as part of a learning opportunity with JPMorgan Chase. It forecasts natural gas prices using seasonal patterns in the data with Linear Regression.

## 📁 Files
- **Natural Gas Price Estimate.py**: Python script with model training, forecasting, and visualization.
- **Nat_Gas.csv**: Input dataset of historical gas prices (with dates).

## 📊 Features
- Preprocessing of time-series data.
- One-hot encoding of monthly seasonality.
- Model training using `sklearn`'s `LinearRegression`.
- Forecast for the next 12 months.
- Price estimate for a custom date (31-Aug-2025).
- Clear Matplotlib visualizations.

## 📦 Requirements
```bash
pip install pandas numpy matplotlib scikit-learn


JP-Morgan-Gas-Price-Forecast/
│
├── Natural Gas Price Estimate.py       # Main Python script for analysis & forecasting
├── Nat_Gas.csv                         # Raw historical natural gas prices
└── README.md                           # Project overview & instructions (to be created)
