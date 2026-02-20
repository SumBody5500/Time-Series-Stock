# Time-Series-Stock
Applied classical time series modelling for financial forecasting using ARIMA
## 1.Advanced Forecasting: 
ARIMAX & Hyperparameter OptimizationStepwise Search: Instead of manually guessing $(p, d, q)$ parameters, I implemented a Stepwise Auto-ARIMA algorithm (AIC-based) to mathematically identify the optimal model lag structure.
## 2.Exogenous Regressors ($X$): 
I upgraded the model from a univariate ARIMA to an ARIMAX by incorporating Trading Volume. This allows the model to factor in market liquidity and volatility when predicting price movements.
## 3.Uncertainty Quantification: 
The forecast includes a 95% Confidence Interval, acknowledging that financial time series are inherently stochastic and providing a risk-adjusted view of the prediction.
