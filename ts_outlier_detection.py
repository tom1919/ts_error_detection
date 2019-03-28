# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:34:47 2019

@author: tommy
"""


# description

# TODO: 
#   - add description
#   - speed up loops by only predicting on last obs for SVM and iforest

#%%
# load libraries

import math
import pandas as pd
import pyper as pr
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from PyAstronomy import pyasl
from sklearn.preprocessing import StandardScaler

#%%
# functions

def loess_res(ts, bass = 0):
    '''
    Fit loess curve using Friedman's Super Smoother and return residuals
    
    Parameters
    ----------
    ts: pandas series
        time series of values 
    bass: int
        smoothing parameter of curve. values up to 10 for more smoothness
        
    Returns
    -------
    list
        series of residuals from loess curve fit
    '''
    
    ts = ts.tolist()
    
    # create a R instance
    r = pr.R(use_pandas = True)
    # pass ts from python to R as Y, and pass bass parameter
    r.assign("Y", ts)
    r.assign("bass", bass)
    
    # fit friedman's super smoother on ts and extract the fitted values
    r("fit = supsmu(x=1:length(Y), y=Y, bass=bass)$y")
    # pass fitted values from r to python
    fit = r.get("fit")
    # residuals from loess fit
    residuals = ts - fit
    return(residuals.tolist())
    
    
def res_bounds(res, lq = 25, uq = 75, llf = 0.95, ulf = 1.2, factor = 2):
    '''
    Use loess curve residuals to identify outliers using upper and lower bounds
    
    Parameters
    ----------
    res: list
        time series of residuals from loess curve fit
    lq: int
        lower quantile
    uq: int
        upper quatile
    ulf: int
        upper limit factor. used as multiplier for upper limit 
    llf: int
        lower limit factor. used as multiplier for lower limit
    factor: int
        used as multipler for both upper and lower limit
        
    Returns
    -------
    list
        Binary series with same length as input TS. A value of 1 indicates the
        corresponding value in TS is an outlier
    '''
    
    residuals = res
    
    # upper and lower percentile of residuals and interquantile range
    up = np.percentile(a = residuals, q = uq)
    lp = np.percentile(a = residuals, q = lq)
    iqr = up - lp
    
    # upper and lower limits calculation
    ul = up + iqr * factor * ulf
    ll = lp - iqr * factor * llf
    
    # Create df and an indicator column of whether residual is beyond limits
    df = pd.DataFrame({"residuals" : residuals})
    df['outlier'] = np.where((df.residuals > ul) | (df.residuals < ll), 1, 0)
    
    return(df.outlier.tolist())
    
    
def res_iforest(features, contamination=.01):
    '''
    use loess curve residuals and isolation forest to identify outliers
    
    Parameters
    ----------
    features: dataframe
        dataframe of features
    contamination: decimal 
        argument in isolation forest for proportion of outliers in data

    Returns
    -------
    list
        Binary series with same length as input TS. A value of -1 indicates the
        corresponding value in TS is an outlier          
    '''
    
    #res = np.asarray(res).reshape(-1,1)
    
    # instantiate and fit iforest on residuals
    model =  IsolationForest(contamination=.01, random_state=88)
    model.fit(features)
   
    # predict on the residuals
    iforest = model.predict(features).tolist()
    
    return(iforest)
    
def res_svm1(features, nu = .01, kernel = "rbf", gamma = .01):
    '''
    use loess curve residuals and 1 class svm to identify outliers
    
    Parameters
    ----------
    features: dataframe
        dataframe of features
    nu: decimal 
        upper bound on the fraction of training errors and a lower bound 
        of the fraction of support vectors, and must be between 0 and 1.
        proportion of outliers expected in data
    kernel: string
        kernel type that svm uses
    gamma: decimal
        parameter of the RBF kernel type and controls the influence of 
        individual training samples - this effects the "smoothness" 
        
    Returns
    -------
    list
        Binary series with same length as input TS. A value of -1 indicates the
        corresponding value in TS is an outlier          
    '''
    
    #res = np.asarray(res).reshape(-1,1)
    
    # instantiate and fit i forest on residuals
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(features)
    
    # predict on the residuals
    svm1 = model.predict(features).tolist()
    
    return(svm1)
    
def res_gesd(res, maxOLs = 15, alpha = 0.05):
    '''
    use loess curve residuals and generalized extreme studentized deviation
    test to identify outliers
    
    Parameters
    ----------
    res: list
        time series of residuals from loess curve fit
        
    maxOLs: int
        max number of outliers to identify using GESD
        
    alpha: decimal
        signifigance level for identifying outlier
         
    Returns
    -------
    list
        Binary series with same length as input TS. A value of 1 indicates the
        corresponding value in TS is an outlier          
    '''
    
    # index values of outliers
    outlier_index = pyasl.generalizedESD(res, maxOLs = maxOLs, alpha = alpha,
                                         fullOutput=False)[1]
    
    # data frame with index values 
    df = pd.DataFrame({'index': list(range(0,len(res)))})
    # create a col to indicate that the index value is an outlier
    df['gesd'] = np.where(df.index.isin(outlier_index), 1, 0)

    return(df.gesd.tolist())
    
def res_dbgesd(res, maxOLs = 8, alpha = 0.05):
    '''
    use loess curve residuals and distance based generalized extreme studentized 
    deviation test to identify outliers
    
    Parameters
    ----------
    res: list
        time series of residuals from loess curve fit
        
    maxOLs: int
        max number of outliers to identify using GESD. Note that the number 
        passed to generalizedESD is actually 2*`maxOLs
        
    alpha: decimal
        signifigance level for identifying outlier
         
    Returns
    -------
    list
        Binary series with same length as input TS. A value of 1 indicates the
        corresponding value in TS is an outlier          
    '''
    
    res = np.asarray(res)
    
    # index values of outliers
    outlier_index = pyasl.pointDistGESD(res, maxOLs = maxOLs, alpha = alpha)[1]
    
    # data frame with index values 
    df = pd.DataFrame({'index': list(range(0,len(res)))})
    # create a col to indicate that the index value is an outlier
    df['dbgesd'] = np.where(df.index.isin(outlier_index), 1, 0)

    return(df.dbgesd.tolist())
      
def impute_error(df, rate=.01, col = 2, lb = .9, ub = 1.1):
    '''
    add errors randomly into a dataframe of TS
    
    Parameters
    ----------
    df : dataframe
        a dataframe of time series in long format
    rate : decimal
        error rate 
    col : int
        col index of where the values are
    lb : decimal
        lower bound on error size
    ub : decimal
        upper bound on error size
    
    Returns
    -------
    dataframe
        same df as input except there's random errors added to it 
    '''

    dirty_df = df.copy()
    
    # num of errors to add. total values * rate
    num_errors = math.ceil(df.shape[0]*rate)
    
    replaced_entries = []
    for i in range(0,num_errors):
        
        # random row 
        rand_row = round(np.random.uniform(0, df.shape[0] -1))
        
        # if value has already been replaced then select another
        while(rand_row in replaced_entries):
            rand_row = round(np.random.uniform(0,df.shape[0] -1))
        
        # modify the value
        dirty_df.iloc[rand_row, col] = dirty_df.iloc[rand_row, col] \
        * np.random.uniform(lb, ub)
        
        # add the replaced entry position to list
        replaced_entries.extend([rand_row])
        
    return dirty_df

def calc_metrics(ts_df, pred_cols) :
    '''
    calculate precision, recall, f1 and accuracy for detecting error value
    
    Parameters
    ----------
    ts_df: dataframe
        has indicator columns for whether row has an actual error and pred error
    pred: list
        strings of names for columns that contain predictions
        
    Returns
    -------
    dataframe
        values for precision, recall, f1 and accuracy. cols identify prediction 
        type
    '''
    
    metrics_df = pd.DataFrame()
    
    for pred in pred_cols:
    
        tp = ts_df.iloc[np.where( (ts_df.error_ind == 1) 
            & (ts_df[pred] == 1) )].shape[0]
        fp = ts_df.iloc[np.where( (ts_df.error_ind == 0) 
            & (ts_df[pred] == 1) )].shape[0]
        tn = ts_df.iloc[np.where( (ts_df.error_ind == 0) 
            & (ts_df[pred] == 0) )].shape[0]
        fn = ts_df.iloc[np.where( (ts_df.error_ind == 1) 
            & (ts_df[pred] == 0) )].shape[0]
        
        prec = tp / (tp+fp)
        recall = tp / (tp+fn)
        f1 = 2*tp / (2*tp+fp+fn)
        acc = (tp+tn) / (tp+fp+tn+fn)
    
        metrics = pd.Series({'Precision':prec, 'Recall': recall, 'F1 Score': f1,
                             'Accuracy':acc})
        
        metrics_df[pred] = metrics
    
    return(metrics_df)
    
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
# close_prices.iloc[100,2] = 88 # appl price on 6/19/18 was 185.69
# close_prices.iloc[150,2] = 888 # appl price on 6/19/18 was 185.69
# # repalce dupes with mean
# close_prices = close_prices.groupby(['Date', 'Symbols']).value.mean()
# close_prices = close_prices.reset_index()
# =============================================================================

# cast date column and sort values
close_prices.Date = pd.to_datetime(close_prices.Date)
close_prices.sort_values(by = ['Symbols', 'Date'], inplace = True)

# create copy of close_prices with errors in it
error_df = impute_error(close_prices)
ts_df = close_prices.copy()
ts_df['error_value'] = error_df.value
ts_df['error_ind'] = np.where(ts_df.value != ts_df.error_value, 1, 0)

#%%
# identify outliers

bounds_lst = []
iforest_lst = []
svm1_lst = []
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
    gesd_lst.extend(res_gesd(res, maxOLs = 15, alpha = 0.05))
    dbgesd_lst.extend(res_dbgesd(res, maxOLs = 8, alpha = 0.05))
                
    print('done with TS: ', cn, '/', len(names))
    cn = cn +1

# create columns for predictions
ts_df['bounds'] = bounds_lst
ts_df['iforest'] = iforest_lst
ts_df['svm1'] = svm1_lst
ts_df['gesd'] = gesd_lst
ts_df['dbgesd'] = dbgesd_lst

# replace -1 with 1 and 0 otherwise
ts_df['iforest'] = np.where(ts_df['iforest'] == -1, 1,0)
ts_df['svm1'] = np.where(ts_df['svm1'] == -1, 1,0)

# calculate metrics for the different predictions
calc_metrics(ts_df, pred_cols = ['bounds', 'iforest', 'svm1', 'gesd', 'dbgesd'])

#%%
# identify outliers while emulating a real time scenerio, stepping through each
# time series one day at a time and just checking the latest value

returns_lst = []
returns_lag_lst = []
res_lst = []
res2_lst = []
res_bounds_lst = []
res_iforest_lst = []
res_svm1_lst = []
res_gesd_lst = []
res_dbgesd_lst = []

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
    res_bounds_lst.extend(zeros)
    res_iforest_lst.extend(zeros)
    res_svm1_lst.extend(zeros)
    res_gesd_lst.extend(zeros)
    res_dbgesd_lst.extend(zeros)
    
    
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
        iforest = res_iforest(features, contamination=.01)
        svm1 = res_svm1(features, nu = .01, kernel = "rbf", gamma = .01)
        gesd = res_gesd(res, maxOLs = 15, alpha = 0.05)
        dbgesd = res_dbgesd(res, maxOLs = 8, alpha = 0.05)
        
        # append last observation to lists
        res_lst.append(res[len(res)-1])
        res2_lst.append(res2[len(res2)-1])
        res_bounds_lst.append(bounds[len(bounds)-1])
        res_iforest_lst.append(iforest[len(iforest)-1])
        res_svm1_lst.append(svm1[len(svm1)-1])
        res_gesd_lst.append(gesd[len(gesd)-1])
        res_dbgesd_lst.append(dbgesd[len(dbgesd)-1])
        
        print('done with step: ', i, 'in TS: ', cn)
        
    print('done with TS: ', cn, '/', len(names))
    cn = cn +1

# create columns for predictions
ts_df['returns'] = returns_lst
ts_df['returns_lag'] = returns_lag_lst
ts_df['res'] = res_lst
ts_df['res2'] = res2_lst
ts_df['bounds'] = res_bounds_lst
ts_df['iforest'] = res_iforest_lst
ts_df['svm1'] = res_svm1_lst
ts_df['gesd'] = res_gesd_lst
ts_df['dbgesd'] = res_dbgesd_lst

# replace -1 with 1 and 0 otherwise
ts_df['iforest'] = np.where(ts_df['iforest'] == -1, 1,0)
ts_df['svm1'] = np.where(ts_df['svm1'] == -1, 1,0)

# calculate metrics for the different predictions
calc_metrics(ts_df, pred_cols = ['bounds', 'iforest', 'svm1', 'gesd', 'dbgesd'])
#%%



























    
    
    
    
    
    
    