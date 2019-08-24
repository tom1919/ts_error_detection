# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:34:47 2019

@author: tommy
"""


# description

# TODO: 
#   - add description
#   - speed up loops by only predicting on last obs where appropriate
#   - add local outlier factor and gaussian envelope
#   - extract anonamoly sores from models as well to use as features

#%%
# load libraries

import os
os.chdir('C:\\Users\\tommy\\Google Drive\\ts_error_detection\\scripts')
import ts_outlier_functions as tsof


import pandas as pd
import pyper as pr
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from PyAstronomy import pyasl
from sklearn.preprocessing import StandardScaler

#%%

#%%
# download stock data and add errors in it

from pandas_datareader import data # get stock data

# download stock data and select only closing prices
tickers = ['^GSPC', 'AAPL', 'GOOGL', 'VZ']
panel_data = data.DataReader(tickers, 'yahoo', '2014-08-25', '2019-01-25')
close_prices = panel_data['Close']

# transform from wide to long format
cols = close_prices.columns.values
close_prices = close_prices.reset_index()
close_prices = pd.melt(close_prices, id_vars = 'Date', value_vars = cols)

# =============================================================================
# # data prep just for practice
# close_prices = pd.concat([close_prices, close_prices]) # create dupes
#
# # repalce dupes with mean
# close_prices = close_prices.groupby(['Date', 'Symbols']).value.mean()
# close_prices = close_prices.reset_index()
# =============================================================================

# cast date column and sort values
close_prices.Date = pd.to_datetime(close_prices.Date)
close_prices.sort_values(by = ['Symbols', 'Date'], inplace = True)

# create copy of close_prices with errors in it
error_df = impute_error(close_prices, lb=.9, ub=1.1)
ts_df = close_prices.copy()
ts_df['error_value'] = error_df.value
ts_df['error_ind'] = np.where(ts_df.value != ts_df.error_value, 1, 0)


#%%



#%%
# identify outliers

bounds_lst = []
iforest_lst = []
svm1_lst = []
lof_lst = []
ee_lst = []
gesd_lst = []
dbgesd_lst = []

scaler = StandardScaler()
names = ts_df.Symbols.unique()
cn = 1   

for name in names :
    
    # filter for 1 asset
    ts = ts_df[ts_df.Symbols == name].error_value
    
    # returns and first lag of it
    ts_returns = ts.pct_change()
    ts_returns.iloc[0] = 0
    ts_returns = ts_returns.replace(np.inf, 0)
    ts_returns = ts_returns.replace(np.NINF, 0)
    ts_returns_lag = ts_returns.shift(1)
    ts_returns_lag.iloc[0] = 0

    # residuals from loess curve fit
    res = loess_res(ts)
    res2 = loess_res(ts, bass = 8)
        
    # create df of features and scale them
    features = pd.DataFrame({'res':res, 'res2':res2, 'returns':ts_returns,
                             'returns_lag': ts_returns_lag})
    scaled = scaler.fit_transform(features)
    features = pd.DataFrame(scaled)
        
    # methods for detecting outliers. each one returns a list indicating if 
    # value is an outlier
    bounds_lst.extend(res_bounds(res))
    iforest_lst.extend(res_iforest(features, contamination=.01))
    svm1_lst.extend(res_svm1(features, nu = .04, kernel = "rbf", 
                             gamma = .01))
    lof_lst.extend(res_lof(features, contamination=0.01, n_neighbors = 20, 
                           score = False))
    ee_lst.extend(res_ee(features, contamination = .01))
    gesd_lst.extend(res_gesd(res, maxOLs = 15, alpha = 0.05))
    dbgesd_lst.extend(res_dbgesd(res, maxOLs = 8, alpha = 0.05))
                
    print('done with TS: ', cn, '/', len(names))
    cn = cn +1

# create columns for predictions
ts_df['bounds'] = bounds_lst
ts_df['iforest'] = iforest_lst
ts_df['svm1'] = svm1_lst
ts_df['lof'] = lof_lst
ts_df['ee'] = ee_lst
ts_df['gesd'] = gesd_lst
ts_df['dbgesd'] = dbgesd_lst

# replace -1 with 1 and 0 otherwise
ts_df['iforest'] = np.where(ts_df['iforest'] == -1, 1,0)
ts_df['svm1'] = np.where(ts_df['svm1'] == -1, 1,0)
ts_df['lof'] = np.where(ts_df['lof'] == -1, 1,0)
ts_df['ee'] = np.where(ts_df['ee'] == -1, 1,0)

# calculate metrics for the different predictions
mets = calc_metrics(ts_df, pred_cols = ['bounds', 'iforest', 'svm1', 'lof', 
                                        'ee', 'gesd', 'dbgesd'])

#%%










#%%
# identify outliers while emulating a real time scenerio, stepping through each
# time series one day at a time and just checking the latest value

returns_lst = []
returns_lag_lst = []
res_lst = []
res2_lst = []
bounds_lst = []
iforest_lst = []
svm1_lst = []
lof_lst = []
ee_lst = []
gesd_lst = []

s_iforest_lst = []
s_svm1_lst = []
s_lof_lst = []
s_ee_lst = []


#res_dbgesd_lst = [] # doesnt work for real time detection

scaler = StandardScaler()
names = ts_df.Symbols.unique()
cn = 1   

for name in names :
    
    # filter for 1 asset
    ts = ts_df[ts_df.Symbols == name].error_value
    
    # returns and first lag of it
    ts_returns = ts.pct_change()
    ts_returns.iloc[0] = 0
    ts_returns = ts_returns.replace(np.inf, 0)
    ts_returns = ts_returns.replace(np.NINF, 0)
    ts_returns_lag = ts_returns.shift(1)
    ts_returns_lag.iloc[0] = 0
    
    # extend returns to list
    returns_lst.extend(ts_returns.tolist())
    returns_lag_lst.extend(ts_returns_lag.tolist())
    
    # first 90 predictions is all 0
    zeros = pd.Series([0]*90).tolist()
    res_lst.extend(zeros)
    res2_lst.extend(zeros)
    bounds_lst.extend(zeros)
    iforest_lst.extend(zeros)
    svm1_lst.extend(zeros)
    lof_lst.extend(zeros)
    ee_lst.extend(zeros)
    gesd_lst.extend(zeros)  
    
    s_iforest_lst.extend(zeros)
    s_svm1_lst.extend(zeros)
    s_lof_lst.extend(zeros)
    s_ee_lst.extend(zeros)
    
    # for each value in TS 91 to the end
    for i in range(90,ts.shape[0]):
        
        # subset from obs 0 to i+1 inclusive for error value
        ts_sub = ts.iloc[0:i+1]
        ts_sub_r = ts_returns[0:i+1]
        ts_sub_rl = ts_returns_lag[0:i+1]
        
        # residuals from loess curve fit
        res = loess_res(ts_sub)
        res2 = loess_res(ts_sub, bass = 8)
        
        # create df of features and scale them
        features = pd.DataFrame({'res':res, 'res2':res2, 'returns':ts_sub_r,
                                 'returns_lag': ts_sub_rl})
        scaled = scaler.fit_transform(features)
        features = pd.DataFrame(scaled)
        
        # methods for detecting outliers. each one returns a list indicating if 
        # value is an outlier
        bounds = res_bounds(res)
        iforest, s_iforest = res_iforest(features, contamination=.01, 
                                         score = True)
        svm1, s_svm1 = res_svm1(features, nu = .01, kernel = "rbf", gamma = .01, 
                        score = True)
        lof, s_lof = res_lof(features, contamination=0.1, n_neighbors = 20, 
                             score = True)
        ee, s_ee = res_ee(features, contamination=0.1, score = True)
        gesd = res_gesd(res, maxOLs = 15, alpha = 0.05)
 
        
        # append last observation to lists
        res_lst.append(res[len(res)-1])
        res2_lst.append(res2[len(res2)-1])
        bounds_lst.append(bounds[len(bounds)-1])
        iforest_lst.append(iforest[len(iforest)-1])
        svm1_lst.append(svm1[len(svm1)-1])
        lof_lst.append(lof[len(lof)-1])
        ee_lst.append(ee[len(ee)-1])
        gesd_lst.append(gesd[len(gesd)-1])
        
        s_iforest_lst.append(s_iforest[len(s_iforest)-1])
        s_svm1_lst.append(s_svm1[len(s_svm1)-1])
        s_lof_lst.append(s_lof[len(s_lof)-1])
        s_ee_lst.append(s_ee[len(s_ee)-1])
        
        
        print('done with step: ', i, 'in TS: ', cn)
        
    print('done with TS: ', cn, '/', len(names))
    cn = cn +1

# create columns for predictions
ts_df['returns'] = returns_lst
ts_df['returns_lag'] = returns_lag_lst
ts_df['res'] = res_lst
ts_df['res2'] = res2_lst
ts_df['bounds'] = bounds_lst
ts_df['iforest'] = iforest_lst
ts_df['svm1'] = svm1_lst
ts_df['lof'] = lof_lst
ts_df['ee'] = ee_lst
ts_df['gesd'] = gesd_lst

ts_df['s_iforest'] = s_iforest_lst
ts_df['s_svm1'] = s_svm1_lst
ts_df['s_lof'] = s_lof_lst
ts_df['s_ee'] = s_ee_lst

# replace -1 with 1 and 0 otherwise
ts_df['iforest'] = np.where(ts_df['iforest'] == -1, 1,0)
ts_df['svm1'] = np.where(ts_df['svm1'] == -1, 1,0)
ts_df['lof'] = np.where(ts_df['lof'] == -1, 1,0)
ts_df['ee'] = np.where(ts_df['ee'] == -1, 1,0)

# calculate metrics for the different predictions
calc_metrics(ts_df, pred_cols = ['bounds', 'iforest', 'svm1', 'lof', 'ee',
                                 'gesd'])
#%%



























    
    
    
    
    
    
    