# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 23:08:37 2019

@author: tommy
"""

# =============================================================================
# - download stock prices
# - add random errors to them
# =============================================================================

#%%


import pandas as pd
import numpy as np
import os
os.chdir('C:\\Users\\tommy\\Google Drive\\ts_error_detection\\scripts')
import ts_outlier_functions as tsof
from pandas_datareader import data # get stock data

# download stock data and select only closing prices
tickers = ['^GSPC', 'AAPL', 'GOOGL', 'VZ', 'BA', 'XOM', 'GE', 'GS', 'HD', 'MCD']
panel_data = data.DataReader(tickers, 'yahoo', '2014-08-25', '2019-04-26')
close_prices = panel_data['Close']

# transform from wide to long format
cols = close_prices.columns.values
close_prices = close_prices.reset_index()
close_prices = pd.melt(close_prices, id_vars = 'Date', value_vars = cols)

# cast date column and sort values
close_prices.Date = pd.to_datetime(close_prices.Date)
close_prices.sort_values(by = ['Symbols', 'Date'], inplace = True)

# create copy of close_prices with errors in it
error_df = tsof.impute_error(close_prices, lb=.9, ub=1.1, rate = .01)
ts_df = close_prices.copy()
ts_df['error_value'] = error_df.value
ts_df['error_ind'] = np.where(ts_df.value != ts_df.error_value, 1, 0)

ts_df.to_csv('C:\\Users\\tommy\\Google Drive\\ts_error_detection\\data\\stock_data.csv')
