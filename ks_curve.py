# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:52:33 2016

@author: evan
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series


def ks_curve(data,show=1):
    '''
    
    :target: to compute the ks-value of the data
    :data: DataFrame, two columns, the first column is the real type(y), 
           the second column is the probability that y equals 1
           
    '''
    
    if not isinstance(data, pd.core.frame.DataFrame):  # class needs a pandas dataframe
        raise AttributeError('input dataset should be a pandas data frame')
    
    data.columns = ['y','prob']
        
    y = data.y
    
    x_axis = np.arange(len(y))/float(len(y))
    PosAll = Series(y).value_counts()[1]
    NegAll = Series(y).value_counts()[0]
    
    data = data.sort_values(by='prob',ascending=False)

    pCumsum = data['y'].cumsum()
    nCumsum = np.arange(len(y))-pCumsum+1
    pCumsumPer = pCumsum/PosAll
    nCumsumPer = nCumsum/NegAll
    
    ks = max(pCumsumPer-nCumsumPer)
    
    if show==1:
        plt.figure(figsize=[6,6])
        plt.plot(x_axis,pCumsumPer,color='red')
        plt.plot(x_axis,nCumsumPer,color='blue')
        plt.title('ks_curve')
        plt.show()
        
    return ks
    
if __name__ == '__main__':
    no_jxl_var = pd.read_csv(r'C:\Users\memedai\Desktop\用新的风险样本验算o2o_no_jxl模型效果_20160831\o2o_no_jxl_score_s.csv',encoding='utf8')
    data = no_jxl_var.ix[:,['Y','credit_score_nojxl']]
    data = data.dropna()
    #data.columns = ['y','prob']
    ks = ks_curve(data,show=1)