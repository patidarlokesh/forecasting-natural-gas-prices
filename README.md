# Natural Gas Price Forecast using Linear Regression

## 🔍 Overview
This project was developed as part of a learning opportunity with JPMorgan Chase. It forecasts natural gas prices using seasonal patterns in the data with Linear Regression.

## 📁 Files
- **Natural Gas Price Estimate.py**: Python script with model training, forecasting, and visualization.
- **Nat_Gas.csv**: Input dataset of historical gas prices (with dates).

## 📊 Features
# Preprocessing of time-series data.
 
      gas_data = pd.read_csv("Nat_Gas.csv")
      gas_data['Dates'] = pd.to_datetime(gas_data['Dates'], format='%m/%d/%y') + MonthEnd(0)
      gas_data.set_index('Dates', inplace=True)

 
# One-hot encoding of monthly seasonality.

    encoder = OneHotEncoder(drop='first', sparse_output=False)
    month_encoded = encoder.fit_transform(gas_data[['Month']])
    month_encoded_df = pd.DataFrame(month_encoded, index=gas_data.index, columns=encoder.get_feature_names_out(['Month']))

# Model training using `sklearn`'s `LinearRegression`.

    model = LinearRegression()
    model.fit(X, y)

# Forecast for the next 12 months.

     last_date = gas_data.index[-1]
     future_dates = pd.date_range(start=last_date + MonthEnd(1), periods=12, freq='M')
     future_data = pd.DataFrame(index=future_dates)
     future_data['Month'] = future_data.index.month
     future_data['Year'] = future_data.index.year
     
     future_month_encoded = encoder.transform(future_data[['Month']])
     future_month_encoded_df = pd.DataFrame(future_month_encoded, index=future_data.index, columns=encoder.get_feature_names_out(['Month']))
     future_X = pd.concat([future_data[['Year']], future_month_encoded_df], axis=1)
     
     future_data['Predicted_Price'] = model.predict(future_X)

# Price estimate for a custom date (31-Aug-2025).
# Clear Matplotlib visualizations.
    plt.plot(...)
    plt.scatter(...)
    plt.annotate(...)


## 📦 Requirements
```bash
pip install pandas numpy matplotlib scikit-learn


JP-Morgan-Gas-Price-Forecast/
│
├── Natural Gas Price Estimate.py       # Main Python script for analysis & forecasting
├── Nat_Gas.csv                         # Raw historical natural gas prices
└── README.md                           # Project overview & instructions (to be created)
