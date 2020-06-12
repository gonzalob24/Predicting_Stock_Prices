# -*- coding: utf-8 -*-

"""
This module will get stock information for the ticker symbols that 
I pass as parameters. 
"""

import pandas as pd
import numpy as np
import pandas_datareader as pdr

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
sns.set_style('whitegrid')  # creates a white grid style when plotting

from pandas import Series, DataFrame
from datetime import datetime
from scipy import stats # to show the pearsonr value

class Portfolio(object):
    """
    stock_list: enter ticker symbols in a list 
    years: int years worth of stock prices
    *args: enter as many moving averages as needed. 5, 10, 20..etc
    """
    
    def __init__(self, stock_list, years, *args):
        """
        stock_list 
        int years worth of stock prices
        enter as many moving averages as needed. 5, 10, 20..etc
        """
        self.stocks = stock_list
        self.years = years
        # amount of moving averages that I will be calculating
        self.mas = args
        self.end_date = datetime.now()
    

    def get_stock_info(self):
        """
        globals takes stock ticker and turns it into a global variable
        so that when I call a stock ticker in main it is a dataframe 
        
        returns stock symbols as globals in a dictionary. Each ticker is a
        DataFrame
        """
        start_date = datetime(self.end_date.year - self.years, self.end_date.month, self.end_date.day)
        
        for stock in self.stocks:
            globals()[stock] = pdr.DataReader(stock, 'yahoo', start_date, self.end_date)
            
        return globals()

    
    def calc_ma(self, stocks_df):
        """
        Put in the portfolio name and calc_ma will calculate each MA and 
        create a new column for each stock in the portfolio.
        
        Parameters
        ----------
        stocks_df : DF object
            enter the globals DF that was returned in main

        Returns
        -------
        None.

        """
        
        for ma in self.mas:
            count = 0
            column_name = '{} Day MA'.format(str(ma))
            for stock in self.stocks:
                temp_name = stocks_df[self.stocks[count]]
                temp_name[column_name] = temp_name['Adj Close'].to_frame().rolling(ma).mean()
                count += 1

    def plot_closing_prices(self, stock1_df, stock2_df, ticker1, ticker2):
        """
        The function will display 4 graphs, it is meant to compare two stocks
        at a time. 
        
        Parameters
        ----------
        stock1_df : global DF
            DataFrame.
        stock2_df : glbal DF
            DataFrame.
        ticker1 : global DF
            string
        ticker2 : global DF
            string
        Returns
        -------
        None.    
        
        graphs are displayed

        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,6))
        fig.subplots_adjust(hspace=0.5)
        ax1.plot(stock1_df['Adj Close'])
        ax2.plot(stock2_df['Adj Close'])
        ax3.plot(stock1_df['Volume'])
        ax4.plot(stock2_df['Volume'])
        
        ax1.set_title('Closing Price for ' + ticker1)
        ax2.set_title('Closing Price for ' + ticker2)
        ax3.set_title('Volume for ' + ticker1)
        ax4.set_title('Volume ' + ticker2)
        
        plt.show()
        
        # Or take a look at each individually
        # tech_stocks_df['AAPL']['Adj Close'].plot(legend=True, figsize=(10,4))
        # tech_stocks_df['AAPL']['Volume'].plot(legend=True, figsize=(10,4))

    def plot_mas(self, *args, stock_symbol_list, stock_df1, stock_df2=None, stock_df3=None, stock_df4=None):
        """
        Will plot MA's for up to 4 stocks. Must include at least 1 stock and 
        1 MA.
        
        Parameters
        ----------
        stock_df1 : global DF from main
            positional DataFrame.
            
        stock_symbol_list : list
            positional list.
        
        *args : MA's 5, 10, 20, 50
            DataFrame symbol.
        
         stock_df2 : global DF from main
            keyworded DataFrame optional.
        
         stock_df3 : global DF from main
            keyworded DataFrame optional.
            
         stock_df4 : global DF from main
            keyworded DataFrame optional.

        Returns
        -------
        None. Plots MA's for the indicated stocks

        """
        list_length = len(stock_symbol_list)
        
        if list_length == 4:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,10))
            fig.subplots_adjust(hspace=0.5)
            if len(args) == 1:
                mas_to_plot = ['Adj Close', str(args[0])  + ' Day MA']
                l1, l2 = ax1.plot(stock_df1[mas_to_plot])
                ax1.legend((l1,l2), (mas_to_plot), loc='upper left', shadow=True)
                ax2.legend((l1,l2), (mas_to_plot), loc='upper left', shadow=True)
                ax3.legend((l1,l2), (mas_to_plot), loc='upper left', shadow=True)
                ax4.legend((l1,l2), (mas_to_plot), loc='upper left', shadow=True)
            elif len(args) == 2:
                mas_to_plot = ['Adj Close', str(args[0])  + ' Day MA', str(args[1]) + ' Day MA']
                l1, l2, l3 = ax1.plot(stock_df1[mas_to_plot])
                ax1.legend((l1,l2, l3), (mas_to_plot), loc='upper left', shadow=True)
                ax2.legend((l1,l2, l3), (mas_to_plot), loc='upper left', shadow=True)
                ax3.legend((l1,l2, l3), (mas_to_plot), loc='upper left', shadow=True)
                ax4.legend((l1,l2, l3), (mas_to_plot), loc='upper left', shadow=True)
            elif len(args) == 3:
                mas_to_plot = ['Adj Close', str(args[0])  + ' Day MA', str(args[1]) + ' Day MA', str(args[2]) + ' Day MA']
                l1, l2, l3, l4 = ax1.plot(stock_df1[mas_to_plot])
                ax1.legend((l1,l2, l3, l4), (mas_to_plot), loc='upper left', shadow=True)
                ax2.legend((l1,l2, l3, l4), (mas_to_plot), loc='upper left', shadow=True)
                ax3.legend((l1,l2, l3, l4), (mas_to_plot), loc='upper left', shadow=True)
                ax4.legend((l1,l2, l3, l4), (mas_to_plot), loc='upper left', shadow=True)
            else: # self.mas == 4
                mas_to_plot = ['Adj Close', str(args[0])  + ' Day MA', str(args[1]) + ' Day MA', str(args[2]) + ' Day MA', str(args[3]) + ' Day MA']
                l1, l2, l3,l4, l5 = ax1.plot(stock_df1[mas_to_plot])
                ax1.legend((l1,l2, l3, l4, l5), (mas_to_plot), loc='upper left', shadow=True)
                ax2.legend((l1,l2, l3, l4, l5), (mas_to_plot), loc='upper left', shadow=True)
                ax3.legend((l1,l2, l3, l4, l5), (mas_to_plot), loc='upper left', shadow=True)
                ax4.legend((l1,l2, l3, l4, l5), (mas_to_plot), loc='upper left', shadow=True)
            
            # ax1.plot(stock_df1[mas_to_plot])
            ax2.plot(stock_df2[mas_to_plot])
            ax3.plot(stock_df3[mas_to_plot])
            ax4.plot(stock_df4[mas_to_plot])
            # setting the x and y labels
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Price')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Price')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Price')
            # set the title for each graph
            ax1.set_title(stock_symbol_list[0])
            ax2.set_title(stock_symbol_list[1])
            ax3.set_title(stock_symbol_list[2])
            ax4.set_title(stock_symbol_list[3])
            
            plt.show()
            
        elif list_length == 3:
            fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(15,10))
            fig.subplots_adjust(hspace=0.5)
            if len(args) == 1:
                mas_to_plot = ['Adj Close', str(args[0])  + ' Day MA']
                l1, l2 = ax1.plot(stock_df1[mas_to_plot])
                ax1.legend((l1,l2), (mas_to_plot), loc='upper left', shadow=True)
                ax2.legend((l1,l2), (mas_to_plot), loc='upper left', shadow=True)
                ax3.legend((l1,l2), (mas_to_plot), loc='upper left', shadow=True)
            elif len(args) == 2:
                mas_to_plot = ['Adj Close', str(args[0])  + ' Day MA', str(args[1]) + ' Day MA']
                l1, l2, l3 = ax1.plot(stock_df1[mas_to_plot])
                ax1.legend((l1,l2, l3), (mas_to_plot), loc='upper left', shadow=True)
                ax2.legend((l1,l2, l3), (mas_to_plot), loc='upper left', shadow=True)
                ax3.legend((l1,l2, l3), (mas_to_plot), loc='upper left', shadow=True)
            elif len(args) == 3:
                mas_to_plot = ['Adj Close', str(args[0])  + ' Day MA', str(args[1]) + ' Day MA', str(args[2]) + ' Day MA']
                l1, l2, l3, l4 = ax1.plot(stock_df1[mas_to_plot])
                ax1.legend((l1,l2, l3, l4), (mas_to_plot), loc='upper left', shadow=True)
                ax2.legend((l1,l2, l3, l4), (mas_to_plot), loc='upper left', shadow=True)
                ax3.legend((l1,l2, l3, l4), (mas_to_plot), loc='upper left', shadow=True)
            else: # self.mas == 4
                mas_to_plot = ['Adj Close', str(args[0])  + ' Day MA', str(args[1]) + ' Day MA', str(args[2]) + ' Day MA', str(args[3]) + ' Day MA']
                l1, l2, l3,l4, l5 = ax1.plot(stock_df1[mas_to_plot])
                ax1.legend((l1,l2, l3, l4, l5), (mas_to_plot), loc='upper left', shadow=True)
                ax2.legend((l1,l2, l3, l4, l5), (mas_to_plot), loc='upper left', shadow=True)
                ax3.legend((l1,l2, l3, l4, l5), (mas_to_plot), loc='upper left', shadow=True)
            
            
            ax2.plot(stock_df2[mas_to_plot])
            ax3.plot(stock_df3[mas_to_plot])
            # setting the x and y labels
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Price')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Price')
            
            # set the title for each graph
            ax1.set_title(stock_symbol_list[0])
            ax2.set_title(stock_symbol_list[1])
            ax3.set_title(stock_symbol_list[2])
            
            plt.show()
            
        elif list_length == 2:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,10))
            fig.subplots_adjust(hspace=0.5)
            if len(args) == 1:
                mas_to_plot = ['Adj Close', str(args[0])  + ' Day MA']
                l1, l2 = ax1.plot(stock_df1[mas_to_plot])
                ax1.legend((l1,l2), (mas_to_plot), loc='upper left', shadow=True)
                ax2.legend((l1,l2), (mas_to_plot), loc='upper left', shadow=True)
            elif len(args) == 2:
                mas_to_plot = ['Adj Close', str(args[0])  + ' Day MA', str(args[1]) + ' Day MA']
                l1, l2, l3 = ax1.plot(stock_df1[mas_to_plot])
                ax1.legend((l1,l2, l3), (mas_to_plot), loc='upper left', shadow=True)
                ax2.legend((l1,l2, l3), (mas_to_plot), loc='upper left', shadow=True)
            elif len(args) == 3:
                mas_to_plot = ['Adj Close', str(args[0])  + ' Day MA', str(args[1]) + ' Day MA', str(args[2]) + ' Day MA']
                l1, l2, l3, l4 = ax1.plot(stock_df1[mas_to_plot])
                ax1.legend((l1,l2, l3, l4), (mas_to_plot), loc='upper left', shadow=True)
                ax2.legend((l1,l2, l3, l4), (mas_to_plot), loc='upper left', shadow=True)
            else: # self.mas == 4
                mas_to_plot = ['Adj Close', str(args[0])  + ' Day MA', str(args[1]) + ' Day MA', str(args[2]) + ' Day MA', str(args[3]) + ' Day MA']
                l1, l2, l3,l4, l5 = ax1.plot(stock_df1[mas_to_plot])
                ax1.legend((l1,l2, l3, l4, l5), (mas_to_plot), loc='upper left', shadow=True)
                ax2.legend((l1,l2, l3, l4, l5), (mas_to_plot), loc='upper left', shadow=True)
            
            ax2.plot(stock_df2[mas_to_plot])
            
            # setting the x and y labels
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Price')
            
            # set the title for each graph
            ax1.set_title(stock_symbol_list[0])
            ax2.set_title(stock_symbol_list[1])
            
            plt.show()
        else: # list_length == 1
            fig, ax1 = plt.subplots(1, 1, figsize=(15,10))
            fig.subplots_adjust(hspace=0.5)
            if len(args) == 1:
                mas_to_plot = ['Adj Close', str(args[0])  + ' Day MA']
                l1, l2 = ax1.plot(stock_df1[mas_to_plot])
                ax1.legend((l1,l2), (mas_to_plot), loc='upper left', shadow=True)
            elif len(args) == 2:
                mas_to_plot = ['Adj Close', str(args[0])  + ' Day MA', str(args[1]) + ' Day MA']
                l1, l2, l3 = ax1.plot(stock_df1[mas_to_plot])
                ax1.legend((l1,l2, l3), (mas_to_plot), loc='upper left', shadow=True)
            elif len(args) == 3:
                mas_to_plot = ['Adj Close', str(args[0])  + ' Day MA', str(args[1]) + ' Day MA', str(args[2]) + ' Day MA']
                l1, l2, l3, l4 = ax1.plot(stock_df1[mas_to_plot])
                ax1.legend((l1,l2, l3, l4), (mas_to_plot), loc='upper left', shadow=True)
            else: # self.mas == 4
                mas_to_plot = ['Adj Close', str(args[0])  + ' Day MA', str(args[1]) + ' Day MA', str(args[2]) + ' Day MA', str(args[3]) + ' Day MA']
                l1, l2, l3,l4, l5 = ax1.plot(stock_df1[mas_to_plot])
                ax1.legend((l1,l2, l3, l4, l5), (mas_to_plot), loc='upper left', shadow=True)
    
            # setting the x and y labels
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')
            
            # set the title for each graph
            ax1.set_title(stock_symbol_list[0])
            
            plt.show()
    
    def calc_daily_returns(self, stocks_df):
        """
        Put in the portfolio name and calulate daily returns and 
        create a new column for each stock in the portfolio.
        
        Parameters
        ----------
        stocks_df : DF from main
            DataFrame that us used in main.

        Returns
        -------
        None.

        """
        count = 0        
        for stock in self.stocks:
            temp_name = stocks_df[self.stocks[count]]
            temp_name['Daily Return'] = temp_name['Adj Close'].pct_change()
            count += 1
    
    def only_daliy_returns(self):
        
        start_date = datetime(self.end_date.year - 1, self.end_date.month, self.end_date.day)
        Adj_Close_df = pdr.DataReader(self.stocks, 'yahoo', start_date, self.end_date)['Adj Close']
        # daily return for each stock
        stock_returns = Adj_Close_df.pct_change()
        
        return stock_returns, Adj_Close_df
    
    def plot_daily_returns(self, stock_symbol_list, stock_df1, stock_df2=None, stock_df3=None, stock_df4=None):
        """
        Function will plit 1 - 4 stocks at a time

        Parameters
        ----------
        stock_symbol_list : TYPE
            DESCRIPTION.
        stock_df1 : TYPE
            DESCRIPTION.
        stock_df2 : TYPE, optional
            DESCRIPTION. The default is None.
        stock_df3 : TYPE, optional
            DESCRIPTION. The default is None.
        stock_df4 : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        list_length = len(stock_symbol_list)
        # colors = ['r', 'b', 'g', 'm' ]
        # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,10))
        # fig.subplots_adjust(hspace=0.5)
        dt_plot = ['Daily Return']
        if list_length == 1:
            fig, (ax1) = plt.subplots(1, 1, figsize=(15,10))
            fig.subplots_adjust(hspace=0.5)
            
            l1 = ax1.plot(stock_df1[dt_plot], color='r', linestyle='--', marker='.', markerfacecolor='blue', markersize=12)
            ax1.legend(l1, (dt_plot), loc='upper left', shadow=True)
            
            # set x and y labels
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Percentage')
            # set the title for each graph
            ax1.set_title(stock_symbol_list[0])
            plt.show()
            
        elif list_length == 2:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,10))
            fig.subplots_adjust(hspace=0.5)
            # leg = ax1.plot(stock_df1[dt_plot])
            l1 = ax1.plot(stock_df1[dt_plot], color='r', linestyle='--', marker='.', markerfacecolor='blue', markersize=12)
            l2 = ax2.plot(stock_df2[dt_plot], color='g', linestyle='--', marker='.', markerfacecolor='blue', markersize=12)
            ax1.legend(l1, (dt_plot), loc='upper left', shadow=True)
            ax2.legend(l2, (dt_plot), loc='upper left', shadow=True)
            
            # ax1.plot(stock_df1[dt_plot], color='r', linestyle='--', marker='.', markerfacecolor='blue', markersize=12)
            # ax2.plot(stock_df2[dt_plot], color='g', linestyle='--', marker='.', markerfacecolor='blue', markersize=12)
            
            # set x and y labels
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Percentage')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Percentage')
            # set the title for each graph
            ax1.set_title(stock_symbol_list[0])
            ax2.set_title(stock_symbol_list[1])
            plt.show()
        
        elif list_length == 3:
            fig, ((ax1, ax2), (ax3)) = plt.subplots(3, 1, figsize=(15,10))
            fig.subplots_adjust(hspace=0.5)
            # leg = ax1.plot(stock_df1[dt_plot])
            l1 = ax1.plot(stock_df1[dt_plot], color='r', linestyle='--', marker='.', markerfacecolor='blue', markersize=12)
            l2 = ax2.plot(stock_df2[dt_plot], color='g', linestyle='--', marker='.', markerfacecolor='blue', markersize=12)
            l3 = ax2.plot(stock_df3[dt_plot], color='m', linestyle='--', marker='.', markerfacecolor='blue', markersize=12)
            ax1.legend(l1, (dt_plot), loc='upper left', shadow=True)
            ax2.legend(l2, (dt_plot), loc='upper left', shadow=True)
            ax3.legend(l3, (dt_plot), loc='upper left', shadow=True)
            
            # set x and y labels
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Percentage')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Percentage')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Percentage')
            # set the title for each graph
            ax1.set_title(stock_symbol_list[0])
            ax2.set_title(stock_symbol_list[1])
            ax3.set_title(stock_symbol_list[2])
            plt.show()
            
        else: # lsit_length == 4
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,10))
            fig.subplots_adjust(hspace=0.5)
            # leg = ax1.plot(stock_df1[dt_plot])
            l1 = ax1.plot(stock_df1[dt_plot], color='r', linestyle='--', marker='.', markerfacecolor='blue', markersize=12)
            l2 = ax2.plot(stock_df2[dt_plot], color='g', linestyle='--', marker='.', markerfacecolor='blue', markersize=12)
            l3 = ax3.plot(stock_df3[dt_plot], color='m', linestyle='--', marker='.', markerfacecolor='blue', markersize=12)
            l4 = ax4.plot(stock_df3[dt_plot], color='k', linestyle='--', marker='.', markerfacecolor='blue', markersize=12)
            ax1.legend(l1, (dt_plot), loc='upper left', shadow=True)
            ax2.legend(l2, (dt_plot), loc='upper left', shadow=True)
            ax3.legend(l3, (dt_plot), loc='upper left', shadow=True)
            ax4.legend(l4, (dt_plot), loc='upper left', shadow=True)
            
            # set x and y labels
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Percentage')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Percentage')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Percentage')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Percentage')
            # set the title for each graph
            ax1.set_title(stock_symbol_list[0])
            ax2.set_title(stock_symbol_list[1])
            ax3.set_title(stock_symbol_list[2])
            ax4.set_title(stock_symbol_list[3])
            plt.show()
            
    def normal_dist_plot(self, stock_symbol_list, stock_df1, stock_df2=None, stock_df3=None, stock_df4=None):
        
        list_length = len(stock_symbol_list)
        dist_plot = 'Daily Return'
        if list_length == 1:
            fig, (ax1) = plt.subplots(1, 1, figsize=(15,10), sharex=True)
            fig.subplots_adjust(hspace=0.5)
            ax1 = sns.distplot(stock_df1[dist_plot].dropna(), bins=50, color='skyblue', ax=ax1).set_title(stock_symbol_list[0])
            skyblue = mpatches.Patch(color='skyblue', label='Best fit line')
            plt.legend(handles=[skyblue])
            plt.show()
        elif list_length == 2:
            fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(15,10), sharex=True)
            fig.subplots_adjust(hspace=0.5)
            ax1 = sns.distplot(stock_df1[dist_plot].dropna(), bins=50, color='skyblue', ax=ax1).set_title(stock_symbol_list[0])
            ax2 = sns.distplot(stock_df2[dist_plot].dropna(), bins=50, color='olive', ax=ax2).set_title(stock_symbol_list[1])
        elif list_length == 3:
            fig, ((ax1, ax2), (ax3)) = plt.subplots(3, 1, figsize=(15,10), sharex=True)
            fig.subplots_adjust(hspace=0.5)
            sns.distplot(stock_df1[dist_plot].dropna(), bins=50, color='skyblue', ax=ax1).set_title(stock_symbol_list[0])
            sns.distplot(stock_df2[dist_plot].dropna(), bins=50, color='olive', ax=ax2).set_title(stock_symbol_list[1])
            sns.distplot(stock_df3[dist_plot].dropna(), bins=50, color='teal', ax3=ax3).set_title(stock_symbol_list[2])

        else: # list_length == 4
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,10), sharex=True)
            fig.subplots_adjust(hspace=0.5)
            sns.distplot(stock_df1[dist_plot].dropna(), bins=50, color='skyblue', ax=ax1).set_title(stock_symbol_list[0])
            sns.distplot(stock_df2[dist_plot].dropna(), bins=50, color='olive', ax=ax2).set_title(stock_symbol_list[1])
            sns.distplot(stock_df3[dist_plot].dropna(), bins=50, color='teal', ax=ax3).set_title(stock_symbol_list[2])
            sns.distplot(stock_df4[dist_plot].dropna(), bins=50, color='blue', ax=ax4).set_title(stock_symbol_list[3])
    
    def plots_compare_stocks(self, stock_symbol_list, stocks_df, plot_type=None):
        """
        This function will allow the user to select the type of plot and 
        up to 4 stocks to compare stock returns and try to capture any
        correlation between stocks. 
        
        Parameters
        ----------
        stock_symbol_list : List
            List of stocks you want to plot and compare.
        stocks_df : DataFrame
            DF stocks returns or closing prices.
        plot_type : TYPE, optional
            Plot type: jointplot, pairplot, pairgrid. The default is None.
        plot_kind_list : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        kind = ['scatter', 'reg' , 'resid' , 'kde' , 'hex']
        
        if plot_type is None and len(stock_symbol_list) == 2:
            sns.jointplot(stock_symbol_list[0], stock_symbol_list[1], stocks_df, kind='reg', color='seagreen').annotate(stats.pearsonr)
        elif plot_type == 'pairplot':
            # default setting for pair plots
            sns.pairplot(stocks_df, size=1.5)
        elif plot_type == 'pairgrid':
            # more control of what graphs to include in the pairplots
            returns_fig = sns.PairGrid(stocks_df.dropna())
            returns_fig.map_upper(plt.scatter, color='purple')
            returns_fig.map_lower(sns.kdeplot, cmap='cool_d')
            returns_fig.map_diag(plt.hist, bins=30)
        elif plot_type == 'heatmap':
            # correlation plot
            sns.heatmap(stocks_df.corr(), annot=True)

    ##### RISK ANALYSIS ########
    
    def basic_risk_analysis(self, returns_df):
        """
        # Now I will quantify the risk of every stock in my portfolio.
        # I will use daily returns and compare them to the expected return with the
        # standar deviation of the daily returns. (A basic approach)"
        
        """
        rets = returns_df.dropna()

        area = np.pi*20   # area used to define the circles of scatter pliot
        
        plt.scatter(rets.mean(), rets.std(), s=area)
        plt.xlabel('Expected Return')
        plt.ylabel('Rik')
        
        # http://matplotlib.org/users/annotations_guid.html
        for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
            plt.annotate(
                label,
                xy = (x, y), xytext = (50, 50),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3, rad=-0.3'))
        return 
    
    def value_at_risk(self, returns_df):
        """
        Value at risk is the amount of money I would be expected to lose (put at risk)
        for a given confidence interval. 
        
        I will first use the bootsrap method. For this method I will calculate the 
        empirical quantiles from a histogram of daily returns. 
        
        The value(s) means that with 95% confidence my worst daily loss should not be more
        than the calculated value
        """
        # Code for one stock
        # sns.distplot(AAPL['Daily Return'].dropna(), bins=100, color = 'purple')
        # get quantiles
        # daily_loss = rets['AAPL'].quantile(0.05)
        # print('With a 95% confidence my worst daily loss would not be more than {:.2}%'.format(-1*daily_loss*100))
        
        rets = returns_df.dropna()
        # for loop to calculate value at rirsk for all stock in portfolio
        print('With 95% confidence my worst daily loss for:')
        for stock in self.stocks:
            daily_loss = rets[stock].quantile(0.05)
            print('\t {} would not be more than {:.2}%'.format(stock, -1*daily_loss*100))

# l1, l2, l3 = ax1.plot(tech_stocks_df['AAPL'][['Adj Close', '10 Day MA', '50 Day MA']])
# ax1.legend((l1, l2, l3), ('Adj Close', '10 Day MA', '50 Day MA'), loc='upper left', shadow=True )

#### Older Code ####
                
# def get_stock_info(stock_list, *args):
#     """
#     grab some data using globals
#     globals takes stock ticker and turns it into a global variable
#     so that when I call a stock ticker in main it is a dataframe 
    
#     Pass in as many ticker symbols as needed
#     """
#     end_date = datetime.now()
#     years = int(input("How many years of data: "))
    
#     start_date = datetime(end_date.year - years, end_date.month, end_date.day)
    
#     for stock in (args or stock_list):
#         globals()[stock] = pdr.DataReader(stock, 'yahoo', start_date, end_date)
        
#     return globals()

# def calc_ma(globs_variable, stock_symbol_list, *args):
#     """
#     Enter the ticker symbols and the number of moving averages you want 
#     to derive
#         Example: AAPL, AMZN, 10, 20, 50...
    
#     will add a new column to each MA to the ticker symbols entered. 
#     """
    
#     for ma in args:
#         count = 0
#         column_name = '{} Day MA'.format(str(ma))
#         for stock in stock_symbol_list:
#             temp_name = globs_variable[stock_symbol_list[count]]
#             temp_name[column_name] = temp_name['Adj Close'].to_frame().rolling(ma).mean()
#             count += 1
#             # AAPL[column_name] = AAPL['Adj Close'].to_frame().rolling(ma).mean()
#     # return globs_variable
        
# Or take a look at each individually
# tech_stocks_df['AAPL']['Adj Close'].plot(legend=True, figsize=(10,4))
# tech_stocks_df['AAPL']['Volume'].plot(legend=True, figsize=(10,4))