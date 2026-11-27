
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
data = pd.read_csv("data.csv")

data["date"] = pd.to_datetime(data["date"])
data = data.sort_values("date")

series = data.set_index("date")["value"]

# Split dataset into training and testing
split = int(len(series) * 0.8)

train = series.iloc[:split]
test = series.iloc[split:]

# Build ARIMA model
model = auto_arima(
    train,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore"
)

print("Best ARIMA Order:", model.order)

# Generate predictions
forecast = model.predict(n_periods=len(test))

# Evaluate model
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))

# Forecast future values
# Refit model using the complete dataset
model.fit(series)

# Predict next 6 months
future_forecast = model.predict(n_periods=6)

future_dates = pd.date_range(
    start=series.index[-1] + pd.DateOffset(months=1),
    periods=6,
    freq="MS"
)

future_results = pd.DataFrame({
    "Date": future_dates,
    "Forecast": future_forecast
})

print("\nNext 6 Months Forecast:")
print(future_results)


# Plot results
plt.figure(figsize=(12, 6))

plt.plot(train.index, train, label="Training Data")
plt.plot(test.index, test, label="Actual Data")
plt.plot(test.index, forecast, label="Predicted Data")

plt.title("ARIMA Time Series Forecasting")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("arima_forecast.png")
plt.show()
