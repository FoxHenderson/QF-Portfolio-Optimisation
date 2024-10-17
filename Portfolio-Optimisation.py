import pandas as pd # tabulated data, excel basically
import matplotlib.pyplot as plt #matplotlib for scatter graphs :)
import matplotlib as mpl
import numpy as np # mathematical operations and arrays
import seaborn as sns # statistical plots, like heat maps or violin charts 
# (lots of charts)
import yfinance as yf # takes stock data from online (dont need to generate your own data, backtesting?)
from scipy.optimize import minimize # just want one function that we use to minimise stuff later on

#example portfolio with 6 assets, aims to determine what % of our portfolio
#we should invest in each ticker, based on data from 01/01/2023 - 01/01/2024
tickers = ['RR.L', 'NWG.L','HSBA.L', 'GSK.L', 'BARC.L', 'LLOY.L'] #.L is for london stock exchange
riskfree=0.01 #risk free rate
data = yf.download(tickers, start="2023-01-01", end="2024-01-01")
data = data["Adj Close"] #Adjusted data at close (yfianance requires capital C!!)

#data.plot(figsize=(10, 5)) #plots the historical adjusted close prices

log_returns = np.log(data/ data.shift(1)) #data.shift shifts the rows by 1 so we can calulate the log returns
#log_returns[tickers].hist(figsize=[18, 12], bins=100)

#plot log returns as histogram
#plt.title("Histogram of Log Returns")
#plt.xlabel("Log Returns")
#plt.ylabel("Frequency")
#plt.show()

def portfolio_performance(weights, log_returns): #weights, how much of each asset we'll put into each stock
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    
    portfolio_returns = np.sum(mean_returns * weights) * 252 #252 trading days in a year
    # = w^T Sigma w
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    
    return portfolio_returns, portfolio_volatility

#riskfree rate being the amount you would earn from interest if your money sat in the bank
#default value for riskfree set as 1% (real rate can be calculated using the difference between
#the interest rate and the yield of a gov bond with the same length, in this case 1 year)

#the sharpe ratio is 
def Portfolio_sharpe(weights, log_returns, riskfree):
    p_returns, p_volatility = portfolio_performance(weights, log_returns)
    return -(p_returns - riskfree) / p_volatility

#generate data for 10k portfolios
num_portfolios = 100_000 #70k portfolios takes like 10 years on my laptop
all_weights = np.zeros((num_portfolios, len(tickers)))
returns = np.zeros(num_portfolios)
volatilities = np.zeros(num_portfolios)
sharpe_ratios = np.zeros(num_portfolios)
#initialise variables for optimal portfolio
maxSharpe = -1
maxWeights = np.random.random(len(tickers))

for i in range(num_portfolios):
    #monte-carlo simulation
    weights = np.random.random(len(tickers)) # could write len(tickers) instead of .columns
    weights /= np.sum(weights) #normalise the weights so |w| =< 1
    
    all_weights[i] = weights
    returns[i], volatilities[i] = portfolio_performance(weights, log_returns)
    sharpe_ratios[i] = -(Portfolio_sharpe(weights, log_returns, riskfree)) #negative because "some maths"
    
    #store the portfolio which delivered the highest sharpe value
    if (maxSharpe < sharpe_ratios[i]):
        maxSharpe = sharpe_ratios[i]
        maxWeights = all_weights[i]

#plot data as scatter graph using matplotlib
plt.figure(figure=(10, 6))
scatter = plt.scatter(volatilities, returns, c=sharpe_ratios, cmap="viridis")
plt.colorbar(scatter, label='Sharpe ratio')
plt.xlabel("Volatility (risk)")
plt.ylabel("Expected returns")
plt.title("Feasible set")
plt.show()

#display optimal portfolio, bar chart with weights of each ticker.
plt.figure(figure=(10, 6))
plt.bar(tickers, maxWeights * 100, color='Black', width=0.4)
plt.xlabel("Stock ticker")
plt.ylabel("Proportion of portfolio (%)")
plt.title(f"Optimal portfolio with sharpe ratio {maxSharpe.round(2)}")
plt.show()