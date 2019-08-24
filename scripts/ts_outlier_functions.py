# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 21:06:05 2019

@author: tommy
"""

#%%
# load libraries

import math
import pandas as pd
import pyper as pr
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from PyAstronomy import pyasl

# create a R instance
r = pr.R(use_pandas = True)

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
    # r = pr.R(use_pandas = True)
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
    
    
def res_iforest(features, contamination=.01, score = False):
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
   
    # predict on the features
    y_pred = model.predict(features).tolist()
    
    if score == False:
        return(y_pred)
    
    scores = model.decision_function(features).tolist()
    return(y_pred, scores)
    
def res_svm1(features, nu = .01, kernel = "rbf", gamma = .01, score = False):
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
    
    # instantiate and fit svm on residuals
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(features)
    
    # predict on the features
    y_pred = model.predict(features).tolist()
    
    if score == False:
        return(y_pred)
    
    scores = model.decision_function(features).tolist()
    return(y_pred, scores)

def res_lof(features, contamination=0.1, n_neighbors = 20, score = False):
    '''
    use loess curve residuals and local outlier factor to identify outliers
    
    Parameters
    ----------
    features: dataframe
        dataframe of features
    contamination: decimal 
        proportion of outliers expected in data
    n_neighbors: int
        number of neighbors to use
    score: boolean
        return binary prediction and outlier scores
        
    Returns
    -------
    list
        Binary series with same length as input TS. A value of -1 indicates the
        corresponding value in TS is an outlier     
    list
        outlier score. series with same length as input TS. only returned if 
        score = True
    '''
    
    #res = np.asarray(res).reshape(-1,1)
    
    # instantiate and predict lof on features
    clf = LocalOutlierFactor(n_neighbors = n_neighbors, 
                             contamination = contamination)
    y_pred = clf.fit_predict(features)
    
    if score == False:
        return(y_pred.tolist())
    
    # outlier scores
    scores = clf.negative_outlier_factor_
    
    return(y_pred.tolist(), scores.tolist())
    
def res_ee(features, contamination=0.1, score = False):
    '''
    use loess curve residuals and elliptic envelope to identify outliers
    
    Parameters
    ----------
    features: dataframe
        dataframe of features
    contamination: decimal 
        proportion of outliers expected in data
    score: boolean
        return binary prediction and outlier scores
        
    Returns
    -------
    list
        Binary series with same length as input TS. A value of -1 indicates the
        corresponding value in TS is an outlier     
    list
        outlier score. series with same length as input TS. only returned if 
        score = True
    '''
    
    #res = np.asarray(res).reshape(-1,1)
    
    # instantiate and predict lof on features
    model = EllipticEnvelope(assume_centered = True, store_precision = False,
                             contamination = .1, random_state = 888)
    
    model.fit(features)
    y_pred =  model.predict(features)
       
    if score == False:
        return(y_pred.tolist())
    
    # outlier scores
    scores = model.decision_function(features)
    
    return(y_pred.tolist(), scores.tolist())

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
        passed to generalizedESD is actually 2*maxOLs
        
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

def calc_metrics(ts_df, pred_cols, beta = .5) :
    '''
    calculate precision, recall, f1 and accuracy for detecting error value
    
    Parameters
    ----------
    ts_df: dataframe
        has indicator columns for whether row has an actual error and pred error
        DF must have column named "error_ind". 0 = no error, 1 = error
    pred: list
        strings of names for columns that contain predictions. 0 = no error, 
        1 = error
    beta: decimal
        weight of recall in f-beta score
        
    Returns
    -------
    dataframe
        rows identify accuracy metrics. cols identify prediction type
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
        beta_sq = beta * beta
        f_beta = (1 + beta_sq) * ((prec * recall) / ((beta_sq * prec) + recall))
        #mcc = ...
    
        metrics = pd.Series({'Precision':prec, 
                             'Recall': recall,
                             'F-Beta Score': f_beta,
                             'F1 Score': f1,
                             'Accuracy':acc})
        
        metrics_df[pred] = metrics
    
    return(metrics_df)