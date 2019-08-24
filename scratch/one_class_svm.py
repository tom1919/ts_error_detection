# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 21:35:05 2019

@author: tommy
"""

# identify outliers while emulating a real time scenerio, stepping through each
# time series one day at a time and just checking the latest value
# calculate performance metrics for classifying each value as an error or not

model = OneClassSVM(nu = .04, kernel = "rbf", gamma = .00008)
scaler = StandardScaler()
svm1_lst = []
names = ts_df.Symbols.unique()
ts_n = 1   

for name in names :
    
    # filter for 1 asset
    ts = ts_df[ts_df.Symbols == name].error_value
    
    # daily returns (percent changes)
    ts_returns = ts.pct_change()
    
    # residuals from rolling 15 day mean
    roll_mean_res = ts - ts.rolling(15).mean()
        
    # first 120 predictions is all 0 bc need some history for model
    zeros = pd.Series([0]*120).tolist()
    svm1_lst.extend(zeros)

    # for each value in TS 121 to the end
    for i in range(120,ts.shape[0]):
        
        # subset from obs 15 to i+1 inclusive 
        ts_sub = ts.iloc[15:i+1]
        ts_sub_ret = ts_returns[15:i+1]
        roll_mean_res_sub = roll_mean_res[15:i+1]
        
        # residuals from loess curve fit
        res = loess_res(ts_sub)
        
        # create df of features and scale them
        features = pd.DataFrame({'res': res, 
                                 'returns': ts_sub_ret,
                                 'roll_mean_res': roll_mean_res_sub
                                })
        scaled = scaler.fit_transform(features)
        features = pd.DataFrame(scaled)
        
        # OC SVM for detecting outliers. returns a list indicating if 
        # value is an outlier
        svm1 = model.fit(features).predict(features).tolist()
        
        # append last observation to lists
        svm1_lst.append(svm1[len(svm1)-1])
        
        print('done with step: ', i, 'in TS: ', ts_n) # about 1200 steps for each of the 39 TS
        
    ts_n = ts_n +1

# create columns for predictions
ts_df['svm1'] = svm1_lst

# replace -1 with 1 and 0 otherwise
ts_df['svm1'] = np.where(ts_df['svm1'] == -1, 1,0)

# calculate metrics for the different predictions
oc_svm_results = calc_metrics(ts_df, pred_cols = ['svm1'])