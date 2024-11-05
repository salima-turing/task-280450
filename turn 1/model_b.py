import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Dummy data generation
np.random.seed(0)
date_range = pd.date_range(start='2020-01-01', periods=120, freq='M')
toy_game_data = np.cumsum(np.random.randn(len(date_range)) * 1000 + 10000)
toy_game_df = pd.DataFrame({'Date': date_range, 'Sales': toy_game_data})
toy_game_df.set_index('Date', inplace=True)

# Plot the historical data
toy_game_df['Sales'].plot(figsize=(10, 6), title="Historical Toy Sales")

# Time series analysis
p = d = q = 1  # ARIMA parameters (you can experiment with these values)
model = ARIMA(toy_game_df['Sales'], order=(p, d, q))
model_fit = model.fit()

# Forecasting
future_periods = 6
forecast_start = len(toy_game_df)
forecast_end = forecast_start + future_periods - 1
forecast = model_fit.predict(start=forecast_start, end=forecast_end, typ='levels')

# Create a dataframe for the forecast
forecast_df = pd.DataFrame(
    {'Date': pd.date_range(start=toy_game_df.index[-1] + pd.DateOffset(1), periods=future_periods, freq='M'),
     'Forecasted_Sales': forecast})
forecast_df.set_index('Date', inplace=True)

# Plotting forecast
toy_game_df['Sales'].plot(figsize=(10, 6), title="Historical Toy Sales with Forecast")
forecast_df['Forecasted_Sales'].plot(color='r', ls='--')

# Calculating RMSE
rmse = np.sqrt(mean_squared_error(toy_game_df['Sales'][-future_periods:], forecast))
print("RMSE:", rmse)
