import matplotlib.pyplot as plt
import numpy as np

def plot_final_forecast(history, actual, forecast, conf_int, ticker):
    plt.figure(figsize=(12, 6))
    
    # Plot recent history
    plt.plot(history.index[-100:], history[-100:], label='Historical Price', color='black')
    
    # Plot the "Ground Truth" (what actually happened)
    plt.plot(actual.index, actual, label='Actual Price', color='blue', alpha=0.6)
    
    # Plot Forecast
    plt.plot(actual.index, forecast, label='ARIMAX Forecast', color='red', linestyle='--')
    
    # Plot Confidence Interval
    plt.fill_between(actual.index, 
                     conf_int[:, 0], 
                     conf_int[:, 1], 
                     color='pink', alpha=0.3, label='95% Confidence Interval')
    
    plt.title(f'{ticker} Price Prediction with External Regressors (Volume)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('arimax_forecast.png')
    plt.show()