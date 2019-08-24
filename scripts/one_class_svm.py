# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 23:47:13 2019

@author: tommy
"""

#%%
# load libraries and data

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import os
os.chdir('C:\\Users\\tommy\\Google Drive\\ts_error_detection\\scripts')
import ts_outlier_functions as tsof

ts_df = pd.read_csv('../data/stock_data.csv', index_col =0)
#%%
# fit loess curve

lowess = sm.nonparametric.lowess
window_n = 51
x = np.arange(1, window_n + 1)
    
def lc_fit(y):
    lc_fit = lowess(y, x, frac = .35)[:,1] # larger frac -> more smooth
    lc_fit1 = lc_fit[len(lc_fit) - 1] # last obs
    return(lc_fit1)
    
# fit loess curve using a rolling window. avoids look ahead bias / future peaking
ts_df['lc_fit'] = ts_df.error_value.rolling(window = window_n, center = False).\
    apply(func = lc_fit, raw = True )

#%%
# create features

# residuals from loess curve fit
ts_df['lc_res'] = ts_df.error_value - ts_df.lc_fit

# daily returns
ts_df['returns'] = ts_df.error_value.pct_change()

# residuals from 15 day rolling mean
ts_df['roll_mean_res'] = ts_df.error_value \
    - ts_df.error_value.rolling(15).mean()
        
# ts_df['returns_sd'] = ts_df.returns.rolling(15).std() # no good

ts_df['lc_res_sd'] = ts_df.lc_res.rolling(15).std()

ts_df['lc_res_mean'] = ts_df.lc_res.rolling(15).mean()

# ts_df['roll_mean_res_sd'] = ts_df.roll_mean_res.rolling(15).std() # no good

feature_names = ['returns', # important
                 #'roll_mean_res', # increase precision reduce recall
                 #'lc_res_mean',
                 'lc_res', # important
                 #'lc_res_sd' # increase recall reduce precision
                 ]


#
# split train and test set
    
names = ts_df.Symbols.unique().tolist()

train_lst = []
remove_lst = []

for name in names:
    ts_df1 = ts_df[ts_df.Symbols == name]
    # first n obs of each asset is not useable bc it's used to fit loess curve
    remove_lst.append(ts_df1[0:65]) # index i to j-1 inclusive
    train_lst.append(ts_df1[65:150]) # index i to j-1 inclusive
    
    
train = pd.concat(train_lst).reset_index()
remove = pd.concat(remove_lst).reset_index()

# test set is the obs with index values that are not in train and remove    
test = ts_df.reset_index()
test = test[~test['index'].isin(train['index'])] 
test = test[~test['index'].isin(remove['index'])] 

#
# create models for each asset and save to list

# lists to store scaler and svm models
scalers_lst = []
svm_lst = []

for name in names:
    
    # instantiate svm model and scaler
    svm = OneClassSVM(nu = .01, kernel = "rbf", gamma = .008)
    scaler = StandardScaler()
    
    train1 = train[train.Symbols == name]\
        .loc[:, feature_names]
    
    # fit scaler and store in list to be used for scaling test set
    scaler_mod = scaler.fit(train1)
    scalers_lst.append(scaler_mod)
    
    train1_scaled = scaler_mod.transform(train1)  
    
    # fit svm model and store in list to used for making predictions on test set
    svm_mod = svm.fit(train1_scaled)
    svm_lst.append(svm_mod)
    
#
# make predictions on test set

i = 0
pred_lst = []
for name in names:
    
    test1 = test[test.Symbols == name]\
        .loc[:, feature_names]
        
    test1_scaled = scalers_lst[i].transform(test1)
    
    pred = svm_lst[i].predict(test1_scaled)
    
    pred_lst.extend(pred)
    
    i = i+1
    
#
# calculate performance metrics

test['pred'] = pred_lst
test['pred'] = np.where(test['pred'] == -1, 1, 0)

tsof.calc_metrics(test, ["pred"], beta = .2)   
# null prec: .0097, recall: 1, fbeta: .010086, f1: .019218, accuracy: .009702









