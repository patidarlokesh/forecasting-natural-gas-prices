#import important librarys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from pandas.tseries.offsets import MonthEnd

# Load and preprocess gas_data
gas_data = pd.read_csv("C:/Users/lokes/OneDrive/Documents/JP Morgan Project/Project 1/Nat_Gas.csv")
gas_data['Dates'] = pd.to_datetime(gas_data['Dates'], format='%m/%d/%y') + MonthEnd(0)
gas_data.set_index('Dates', inplace=True)

# Add seasonal features
gas_data['Month'] = gas_data.index.month
gas_data['Year'] = gas_data.index.year

# One-hot encode month
encoder = OneHotEncoder(drop='first', sparse_output=False)
month_encoded = encoder.fit_transform(gas_data[['Month']])
month_encoded_df = pd.DataFrame(month_encoded, index=gas_data.index, columns=encoder.get_feature_names_out(['Month']))

# Prepare features and target
X = pd.concat([gas_data[['Year']], month_encoded_df], axis=1)
y = gas_data['Prices']

# Train the model to learn price patterns
model = LinearRegression()
model.fit(X, y)

# Forecast /Future Outlook next 12 months
last_date = gas_data.index[-1]
future_dates = pd.date_range(start=last_date + MonthEnd(1), periods=12, freq='M')
future_data = pd.DataFrame(index=future_dates)
future_data['Month'] = future_data.index.month
future_data['Year'] = future_data.index.year

# Encode future months
future_month_encoded = encoder.transform(future_data[['Month']])
future_month_encoded_df = pd.DataFrame(future_month_encoded, index=future_data.index, columns=encoder.get_feature_names_out(['Month']))
future_X = pd.concat([future_data[['Year']], future_month_encoded_df], axis=1)

# Predict future prices
future_data['Predicted_Price'] = model.predict(future_X)

# --- Define function to estimate price ---
def get_price_estimate(input_date):
    date = pd.to_datetime(input_date) + MonthEnd(0)
    month = date.month
    year = date.year
    encoded_month = encoder.transform(pd.DataFrame([[month]], columns=['Month']))
    input_X = pd.DataFrame(np.concatenate([[year], encoded_month[0]]).reshape(1, -1),
                           columns=['Year'] + list(encoder.get_feature_names_out(['Month'])))
    predicted_price = model.predict(input_X)[0]
    return round(predicted_price, 2)

# Estimate price for 31-Aug-2025
highlight_date = pd.to_datetime('2025-08-31') + MonthEnd(0)
highlight_price = get_price_estimate(highlight_date)

# Combine for plotting
combined_data = pd.concat([gas_data[['Prices']], future_data[['Predicted_Price']].rename(columns={'Predicted_Price': 'Prices'})])

# --- Plot ---
plt.figure(figsize=(14, 6))
plt.plot(gas_data.index, gas_data['Prices'], label='Historical Prices', marker='o')
plt.plot(future_data.index, future_data['Predicted_Price'], label='Forecasted Prices', marker='x', linestyle='--', color='orange')

# Highlight 31-Aug-2025
plt.scatter(highlight_date, highlight_price, color='red', s=100, zorder=5, label='31-Aug-2025 Forecast')
plt.annotate(f"{highlight_price:.2f}", (highlight_date, highlight_price),
             textcoords="offset points", xytext=(-10, 10), ha='center', color='red', fontsize=10)

plt.title('Natural Gas Prices (Historical + 1-Year Forecast)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print estimated price
print(f"Estimated price on {highlight_date.strftime('%d-%b-%Y')}: {highlight_price}")
