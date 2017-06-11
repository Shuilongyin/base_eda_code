# -*- coding: utf-8 -*-
"""
Created on Thu May 26 18:31:03 2016

@author: memedai
target: calcuate the statistical index of feature
"""

from pandas import DataFrame
import pandas as pd
import numpy as np
import scipy as sp

def calcChi2(mydat):
    """
    :calcuate the chi2 of features
    mydat: pd.DataFrame
    note: X must be discrete
    
    """
    columns = mydat.columns.difference(['Y'])
    chi2Dict={}
    for column in columns:
        columnContingency = np.array(pd.crosstab(mydat.Y,mydat.ix[:,column],aggfunc=[len]))
        columnChi2Tab = sp.stats.chi2_contingency(columnContingency)
        chi2Dict[column]=[columnChi2Tab[0],columnChi2Tab[1]]
    chi2Frame = DataFrame(chi2Dict,index=['chi2','chi2_p'])
    return chi2Frame    
    
if __name__=='__main__':
    data = pd.read_csv(r'D:\getui_integ_0412.csv',encoding='gbk')
    fiData = DataFrame(data.fillna(value=0),columns=['年龄','性别','大学生','美食','外卖','电影','p2p','手机价格','Y'])
    d = calcChi2(fiData)
    
