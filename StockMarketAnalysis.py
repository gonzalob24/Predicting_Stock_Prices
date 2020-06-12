#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

I created this program to put together a lot of the things that I have 
learned about Python, numpy, pandas, plotting, and machine learning. For now, the program
will only use Monte Carlo Simulations to try and predict stock prices. 

I will continue working on this projoct and overtime it will have different 
tools to help predict stock prices. 

My goal is to build the program using object oriented programming. 

These are some of the questions that helped me get started on this project.


- WHAT WAS THE CHANGE IN PRICE OF THE STOCK OVER TIME?
- WHAT WAS THE DAILY RETURN OF THE STOCK ON AVERAGE?
- WHAT WAS THE MOVING AVERAGE OF THE VAROIS STOCK?
- WHAT WAS THE CORRELATION BETWEEN DIFFERENT STOCKS CLOSING PRICE?
- WHAT WAS THE CORRELATION B/W DIFFERENT STOCKS DAILY RETURN?
- HOW MUCH VALUE DO WE PUT AT RISK BY INVESTING IN A PARTICULAR STOCK?
- HOW CAN BE ATTEMPT TO PREDICT FUTURE STOCK BEHAVIOR?

"""

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')  # creates a white grid style when plotting

from getStockInfo import*
from pandas import Series, DataFrame
import pandas_datareader as pdr
from datetime import datetime

# list of stocks that I will be analyzing.
stock_list = ['AAPL', 'NFLX', 'MSFT', 'AMZN']

tech_portfolio = Portfolio(stock_list, 3, 5, 10, 20, 50)

tech_stocks_df = tech_portfolio.get_stock_info()

# input globs because it has stock information from stock list
tech_portfolio.calc_ma(tech_stocks_df)

# Call some ticker symbols to see how the data looks
tech_stocks_df['AAPL'].head()
tech_stocks_df['NFLX'].head()

# How some of descriptive statistics look
tech_stocks_df['AAPL'].describe()
# Look at the number columns and names
tech_stocks_df['AAPL'].info()


###### For plotting think about creating a separate module #### LATER
# historical view of closing price and volume side by side
tech_portfolio.plot_closing_prices(tech_stocks_df['AAPL'], tech_stocks_df['MSFT'], 'AAPL', 'MSFT')

# Visualize the moving averages
tech_portfolio.plot_mas(5, 20, 10, 50, stock_symbol_list =['AAPL'], stock_df1=tech_stocks_df['AAPL'])

# Calculate daily returns for each stock in my list 
tech_portfolio.calc_daily_returns(tech_stocks_df)

# visualize the daily returns max 4 stocks at a time
tech_portfolio.plot_daily_returns(['AAPL', 'NFLX', 'MSFT', 'AMZN'], stock_df1=tech_stocks_df['AAPL'], stock_df2=tech_stocks_df['NFLX'], stock_df3=tech_stocks_df['MSFT'], stock_df4=tech_stocks_df['AMZN'])

# Normal distribution plot of daily returns
tech_portfolio.normal_dist_plot(['AAPL', 'NFLX'], stock_df1=tech_stocks_df['AAPL'], stock_df2=tech_stocks_df['NFLX'])
tech_portfolio.normal_dist_plot(['AAPL'], stock_df1=tech_stocks_df['AAPL'])

tech_portfolio.normal_dist_plot(['AAPL', 'NFLX', 'MSFT', 'AMZN'], stock_df1=tech_stocks_df['AAPL'], stock_df2=tech_stocks_df['NFLX'], stock_df3=tech_stocks_df['MSFT'], stock_df4=tech_stocks_df['AMZN'])


# New DF only for Adj Close for each stock used to calculate daily return
# for each stock. This will make it easier to visualize only the returns
# for each stock or when comparing to other stocks in my portfolio. 
stock_returns_df, closing_price_df= tech_portfolio.only_daliy_returns()
stock_returns_df.head()
closing_price_df.head()

tech_portfolio.plots_compare_stocks(['NFLX', 'NFLX'], stock_returns_df)
tech_portfolio.plots_compare_stocks(['NFLX', 'AAPL'], closing_price_df)
tech_portfolio.plots_compare_stocks(stock_list, stock_returns_df, plot_type='pairgrid')
tech_portfolio.plots_compare_stocks(stock_list, stock_returns_df, plot_type='pairplot')
tech_portfolio.plots_compare_stocks(stock_list, closing_price_df, plot_type='pairplot')

##############################################   
#                                            #
#            RISK ANALYSIS                   #
#                                            #
##############################################  

# From looking at the graph I would want one that gives a strong expected return
# and a lower risk  
tech_portfolio.basic_risk_analysis(stock_returns_df)

##########   VALUE AT RISK   #################   

tech_portfolio.value_at_risk(stock_returns_df)




##############################################   
#                                            #
#                OLD CODE                    #
#                                            #
##############################################   

# list of stocks that I will be analyzing
# tech_stocks = ['AAPL', 'NFLX', 'MSFT', 'AMZN']

# end_date = datetime.now()
# start_date = datetime(end_date.year - 1, end_date.month, end_date.day)

# grab some data using globals
# globals takes stock ticker and turns it into a global variable
# so that when I call a stock ticker it is a dataframe 
# for stock in tech_stocks:
    # globals()[stock] = pdr.DataReader(stock, 'yahoo', start_date, end_date)
    

# I can use the ticker name as a DF due to using globals
# AAPL.head()
# NFLX.head()

# AAPL.describe()
# AAPL.info()

# historical view of closing price
# fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
# ax1.plot(AAPL['Adj Close'])
# ax2.plot(AAPL['Volume'])

# ax1.set_title('Closing Price')
# ax2.set_title('Volume')

# plt.show()


# AAPL['Adj Close'].plot(legend=True, figsize=(10,4))
# AAPL['Volume'].plot(legend=True, figsize=(10,4))


# Calculatiing the moving averages
# MA's help smooth out price action but reducing noise
# trend-following, lagging indicator b/c it is based on past prices

# ma_day = [10, 20 ,50]

# for ma in ma_day:
#     column_name = '{} Day MA'.format(str(ma))
    
#     AAPL[column_name] = AAPL['Adj Close'].to_frame().rolling(ma).mean()

# AAPL[['Adj Close', '10 Day MA', '20 Day MA', '50 Day MA']].plot(subplots = False, figsize=(10,4))

# daily returns for AAPL
# AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()
# AAPL['Daily Return'].plot(figsize=(10,4), legend=True, linestyle='--', marker='o')


# sns.distplot(tech_stocks_df['AAPL']['Daily Return'].dropna(), bins=100, color='blue')
# # just the histogram
# tech_stocks_df['AAPL']['Daily Return'].hist(bins=100)


# # new DF for adj close for each stock
# closing_df = pdr.DataReader(tech_stocks, 'yahoo', start_date, end_date)['Adj Close']

# closing_df.head()

# # daily return for each stock
# stock_returns = closing_df.pct_change()
# stock_returns.head()

#############################################################
# comparing NFLX to itself
# sns.jointplot('NFLX', 'NFLX', stock_returns_df, kind='scatter', color='seagreen')

# # comparing two stocks
# from scipy import stats # to show the pearsonr value
# # # gives a sense of how correlated the daily percentage returns are.
# sns.jointplot('NFLX', 'MSFT', stock_returns_df, kind='reg', color='blue').annotate(stats.pearsonr)

# # pairplots to compare stocks. Will pair plots work with more than 4 stocks?
# sns.pairplot(stock_returns_df, size=2)

# # more control of what graphs to include in the pairplots
# returns_fig = sns.PairGrid(stock_returns_df.dropna())
# returns_fig.map_upper(plt.scatter, color='purple')
# returns_fig.map_lower(sns.kdeplot, cmap='cool_d')
# returns_fig.map_diag(plt.hist, bins=30)


# # closing prices
# returns_fig = sns.PairGrid(closing_df)
# returns_fig.map_upper(plt.scatter, color='purple')
# returns_fig.map_lower(sns.kdeplot, cmap='cool_d')
# returns_fig.map_diag(plt.hist, bins=30)

# # correlation plot
# sns.heatmap(stock_returns_df.corr(), annot=True)
# # on closing prices
# sns.heatmap(closing_price_df.corr(), annot=True)
########################################################

#### RISK ANALYSIS ####

# No I will look analyze the risk of the stocks to quantify risk.
# I will use daily returns and compare them to the expected return with the
# standar deviation of the daily returns. (A basic approach)

# daily expected returns
# rets = stock_returns_df.dropna()

# area = np.pi*20   # area used to define the circles of scatter pliot

# plt.scatter(rets.mean(), rets.std(), s=area)
# plt.xlabel('Expected Return')
# plt.ylabel('Rik')

# # http://matplotlib.org/users/annotations_guid.html
# for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
#     plt.annotate(
#         label,
#         xy = (x, y), xytext = (50, 50),
#         textcoords = 'offset points', ha = 'right', va = 'bottom',
#         arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3, rad=-0.3'))

# From looking at the graph I would want one that gives a strong expected return
# and a lower risk   


# Value at risk is the amount of money I would be expected to lose (put at risk)
# for a given confidence interval. 

# I will first use the bootsrap method. For this method I will calculate the 
# empirical quantiles from a histogram of daily returns. 

# sns.distplot(tech_stocks_df['AAPL']['Daily Return'].dropna(), bins=100, color = 'purple')

# # get quantiles
# daily_loss = rets['AAPL'].quantile(0.05)

# # this value means that with 95% confidence my worst dail loss wold not be more
# # than this value. 

# print('With a 95% confidence my worst daily loss would not be more than {:.2}%'.format(-1*daily_loss*100))

# print('With 95% confidence my worst daily loss for:')
# for stock in tech_stocks:
#     daily_loss = rets[stock].quantile(0.05)
#     print('\t {} would not be more than {:.2}%'.format(stock, -1*daily_loss*100))



# Monte Carlo Simulations
# I will run market simulations with different conditions
# Then I will calculate a portfolio loss for each trial. Then I will sue the 
# aggregation of all the simulations to estabilish how risky the stocks are

# I am going to use the Geometric Brownian Motion (GBM), which is known as the
# Markov Process. The stocks will follow a random walk and be consistent
# with the weak form of the efficient market hypothesis (EMH). Which means that
# past price info. is already incorporated and the next price movement is 
# conditionaly independent of past price movements.

# This boils down to that past info. on the price of a stock is independent 
# of where the stock price will go in the future. As a result I can't exactly 
# predict future prices solely based on the previous price of a stock.


# Set my time horizon
days = 365
# delta for change in time
dlt = 1/days

# mu from NFLX expectd returns
mu = rets.mean()['NFLX']

# sigma is the standard deviation from GBM equation grab from expected daiyl returns 
sigma = rets.std()['NFLX']

# Monte carlo simulation finction
def monte_carlo(start_price, days, mu, sigma):
    """
    starting stock price
    days of simulation
    mu
    sigma
    returns simulated price array
    """
    # price array filled with one
    price = np.zeros(days)
    price[0] = start_price # set my starting price
    
    # Shock and Drift
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    # Run price array for number of days
    for day in range(1,days):
        
        # Calculate Shock
        shock[day] = np.random.normal(loc=mu * dlt, scale=sigma * np.sqrt(dlt))
        # Calculate Drift
        drift[day] = mu * dlt
        # Calculate Price
        price[day] = price[day-1] + (price[day-1] * (drift[day] + shock[day]))
        
    return price


# Get start price from stock_name.head()
start_price = NFLX['Open'][0]
for run in range(100):
    plt.plot(monte_carlo(start_price,days,mu,sigma))  
plt.xlabel("Days")
plt.ylabel("Price")  
plt.title('Monte Carlo Analysis for Netflix')

runs = 10000

simulations = np.zeros(runs)

for run in range(runs):
    # grab the end points
    simulations[run] = monte_carlo(start_price, days, mu, sigma)[days-1]


q = np.percentile(simulations, 1) # 1% meaning 99% of values fit in output


"""
Below will generate a hist of the stock prices.
the starting price and ending mean final price seems to be on point with 
one my previous plots, the one where I plotted expected returns with standard deviation. Form the plot I can see that NFLX's expected return was low,
closerto 0%. As a result from the monte carlo simulation I can see that the 
Mean final price did not increase much. 
"""

plt.hist(simulations, bins=200)
# starting price
plt.figtext(0.6, 0.8, s='Start price: ${:.2f}'.format(start_price))
# mean ending price
plt.figtext(0.6, 0.7, 'Mean final price: ${:.2f}'.format(simulations.mean()))

# variance of the price (w/n 99% confidence interval)
plt.figtext(0.6, 0.6, 'VaR(0.99): ${:.2f}'.format(start_price - q,))

# Display 1% quantile, value at risk. 99% of the time the amount of money
# at most that I will lose is $20.28.
plt.figtext(0.15, 0.6, 'q(0.99): ${:2f}'.format(q))

# plot a line at the 1% quartile result
plt.axvline(x=q, linewidth=4, color='r')

# Title
plt.title('Final price distribution for NFLX after {} days'.format(days), weight='bold')



"""
Questions to think about later
1.) Estimate the values at risk using both methods I learned in this project for a stock not related to technology.

2.) Build a practice portfolio and see how well I can predict risk values with real stock information!

3.) Look further into correlation of two stocks and see if that gives me any insight into future possible stock prices.

"""



































