import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pmdarima import auto_arima

# Dummy data for sales of a toy over the past 12 months
np.random.seed(0)
date_range = pd.date_range(start='2023-01-01', periods=12, freq='MS')
sales_data = np.cumsum(np.random.randint(low=100, high=200, size=12))
data = pd.DataFrame({'Date': date_range, 'Sales': sales_data})
data = data.set_index('Date')


def predict_sales(data, forecast_period=3):
    # Perform time series analysis and forecast future sales

    # Step 1: Define the p,d,q parameters using auto_arima
    stepwise_fit = auto_arima(data['Sales'],
                              start_p=1, start_q=1,
                              max_p=3, max_q=3, m=12,
                              start_P=0, seasonal=True,
                              d=1, D=1, trace=True,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)

    # Step 2: Summarize the model
    print(stepwise_fit.summary())

    # Step 3: Make predictions
    n_periods = forecast_period
    fc, confint = stepwise_fit.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = pd.date_range(start=data.index[-1] + pd.DateOffset(1), periods=n_periods, freq=data.index.freq)

    # step 4: create a Dataframe for the forecast
    forecast_set = pd.DataFrame(fc, index=index_of_fc, columns=['Forecast'])
    lower_bound = pd.DataFrame(confint[:, 0], index=index_of_fc, columns=['lower_bound'])
    upper_bound = pd.DataFrame(confint[:, 1], index=index_of_fc, columns=['upper_bound'])
    forecast_set = pd.concat([forecast_set, lower_bound, upper_bound], axis=1)

    return forecast_set


# Display the dummy data
print("Historical Sales Data:")
print(data)

# Predict future sales
forecast = predict_sales(data, forecast_period=4)

print("\nForecasted Sales:")
print(forecast)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Sales'], label='Historical Sales')
plt.plot(forecast.index, forecast['Forecast'], label='Forecast', color='red')
plt.fill_between(forecast.index,
                 forecast['lower_bound'],
                 forecast['upper_bound'], color='k', alpha=.15)
plt.legend()
plt.show()
