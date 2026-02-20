import yfinance as yf
import pandas as pd
from pmdarima import auto_arima
import visuals


ticker = "NVDA"
data = yf.download(ticker, period="2y", interval="1d")

# Prepare Data
df = data[['Close', 'Volume']].copy()
df = df.asfreq('B').ffill()

# Split into Train and "Future" placeholders for Exogenous variables
train = df.iloc[:-30]
test = df.iloc[-30:] # We'll use the last 30 days to "validate"

# Auto-ARIMA: Finding the Optimal (p, d, q)
# We pass 'Volume' as the exogenous variable (X)
print(f"Searching for optimal parameters for {ticker}...")
auto_model = auto_arima(train['Close'], 
                        X=train[['Volume']], 
                        seasonal=False,   # Stocks rarely have clean 365-day seasonality
                        stepwise=True, 
                        suppress_warnings=True, 
                        error_action="ignore")

print(f"Best Model Found: {auto_model.summary().tables[0]}")

# Forecast next 30 days
# To forecast the future, ARIMA needs to know what the 'Volume' will be.
# For this portfolio piece, we use the actual recent volume as a "perfect foresight" proxy,
# or you could use a rolling average of volume.
forecast_vol = test[['Volume']] 
forecast_series, conf_int = auto_model.predict(n_periods=30, 
                                               X=forecast_vol, 
                                               return_conf_int=True)

# Plot with Confidence Intervals
visuals.plot_final_forecast(train['Close'], test['Close'], forecast_series, conf_int, ticker)