import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import numpy as np

# Sample historical data (replace this with your actual dataset)
data = {
    'Date': ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01', '2022-06-01', '2022-07-01'],
    'Ticket_Sales': [1000, 1200, 1500, 1800, 2000, 1600, 1400]
}
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')


def is_stationary(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    for key, value in result[4].items():
        print('Critical Value (%s): %f' % (key, value))
    # p-value > 0.05: Fail to reject the null hypothesis (Non-stationary)
    return result[1] > 0.05


def predict_sales(data, order=(5, 1, 0)):
    # Make the time series stationary
    if not is_stationary(data):
        data = data.diff().dropna()

    model = ARIMA(data, order=order)
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=3)[0]
    return forecast


# Check for stationarity
print("Is stationary:", is_stationary(df['Ticket_Sales']))

# Predict future sales
predicted_sales = predict_sales(df['Ticket_Sales'])

# Visualize results
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Ticket_Sales'], label='Historical Sales')
plt.plot(df.index[-1] + pd.DateOffset(months=1), predicted_sales, label='Predicted Sales', color='red')
plt.legend()
plt.show()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(df['Ticket_Sales'][-len(predicted_sales):], predicted_sales))
print("RMSE:", rmse)
