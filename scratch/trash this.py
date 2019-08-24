# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 23:47:13 2019

@author: tommy
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

ts_df = pd.read_csv('C:\\Users\\tommy\\Google Drive\\ts_error_detection\\data\\stock_data.csv', index_col =0)


appl = ts_df[ts_df.Symbols == 'AAPL']
y = appl.value
x = np.arange(1,1177)

lowess = sm.nonparametric.lowess

z = lowess(y.iloc[1000:1050],x[1000:1050], frac = .15)[:,1]
#z = lowess(y,x, frac = .1)[:,1]



plt.plot(y.iloc[1000:1050].reset_index().iloc[:,1])
plt.plot(z)
plt.show()

#%%

ts_df['returns'] = ts_df.error_value.pct_change()

ts_df['roll_mean_res'] = ts_df.error_value - ts_df.error_value.rolling(15).mean()

lc_res_lst = []


ts_error_value = ts_df.error_value
lowess = sm.nonparametric.lowess

x = np.arange(1,102) # 1 - 101 inclusive
y = ts_error_value[2:103].tolist() # index 0-100 inclusive
lc_fit = lowess(y, x, frac = .1)[:,1]
lc_res = y - lc_fit
lc_res_lst.extend(lc_res)

for i in range(1, ts_df.shape[0]):
    
    y = ts_error_value[i:i+101].tolist()
    lc_fit = lowess(y, x, frac = .1)[:, 1]
    
    lc_res = y[len(y) - 1] - lc_fit[len(lc_fit) - 1]
    lc_res_lst.append(lc_res)    
    print(i)
    
#%%
    
for i in range(1,500):
    print(i)    
    
    
    
    
def lc_res(y):
    lc_fit = lowess(y, x, frac = .15)[:,1]
    lc_res1 = y[len(y) - 1] - lc_fit[len(lc_fit) - 1]
    return(lc_res1)
    
foo = ts_error_value.rolling(window = 51, center = False).\
    apply(func = lc_res, raw = True )
    

    
    
    
    
    

