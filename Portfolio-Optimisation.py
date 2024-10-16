#%%

import pandas as pd # tabulated data, excel basically
import matplotlib as plt # plotting line/scatter graphs
import numpy as np # mathematical operations and arrays
import seaborn as sns # statistical plots, like heat maps or violin charts 
# (lots of charts)
import yfinance as yf # takes stock data from online (dont need to generate your own data, backtesting?)
from scipy.optimize import minimize # just want one function that we use to minimise stuff later on

tickers = ['RR.L',  'NWG.L','HSBA.L', 'GSK.L', 'BARC.L', 'LLOY.L'] #.L is for london stock exchange
data = yf.download(tickers, start="2023-01-01", end="2024-01-01")
data = data["Adj close"] #Adjusted data at close
data.plot() #plots the historical adjusted close prices
#print(data.head())

log_returns = np.log(data/ data.shift(1)) #data.shift shifts the rows by 1 so we can calulate the log returns
log_returns.hist(figsize=[18, 12], bins=100)

def portfolio_performance(weights, log_returns): #weights, how much of each asset we'll put into each stock
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    
    portfolio_returns = np.sum(mean_returns * weights) * 252 #252 trading days in a year
    # = w^T Sigma w
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    
    return portfolio_returns, portfolio_volatility

def Portfolio_sharpe(weights, log_returns, riskfree):
    p_returns, p_volatility = portfolio_performance(weights, log_returns)
    return -(p_returns - riskfree) / p_volatility

num_portfolios = 70_000
all_weights = np.zeros(((num_portfolios, len(tickers))))
returns = np.zeros(num_portfolios)
volatilities = np.zeros(num_portfolios)
sharpe_ratios = np.zeros(num_portfolios)

for i in range(num_portfolios):
    weights = np.random.random(len(log_returns.columns)) # could write len(tickers) instead of .columns
    weights /= np.sum(weights)
    
    all_weights[1, i] = weights
    returns[i], volatilities[i] = portfolio_performance(weights, log_returns)
    sharpe_ratios[i] = -(portfolio_performance(weights, log_returns))
    
plt.figure(figure=(10, 0))
plt.scatter(volatilities, returns, c=sharpe_ratios, cmap="viridis")
plt.xlabel("Volatility (risk)")
plt.ylabel("Expected returns")
plt.title("Feasible set")
plt.show()