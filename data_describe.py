"""
author:evan
date:2016-05-22
target:data_describe

"""
import pandas as pd
import numpy as np
from pandas import DataFrame


def calcNull(mydat):
    """
    :calcuate the rate of null in the data set.
    :mydat,pd.DataFrame
    
    """
    columns = mydat.columns
    sumCount = len(mydat.index)
    colsDf = pd.DataFrame({'tmp':['tmp',np.nan]})
    for col in columns:
        nullCountDict = {}
        nullNum = pd.isnull(df[col]).sum()
        nullCountDict[col] = [nullNum,(nullNum*1.0/sumCount)]
        colDf = pd.DataFrame(nullCountDict)
        colsDf = colsDf.join(colDf)
    colsDf = colsDf.rename(index={0:'nullCount',1:'nullRate'}).drop('tmp',axis=1)
    return colsDf

def calcConcentric(mydat):
    """
    :calcuate the concentration rate
    :mydat,pd.DataFrame
    
    """   
    columns = mydat.columns
    sumCount = len(mydat.index)
    colsDf = DataFrame({'tmp':['tmp',np.nan,np.nan]})
    for col in columns:
        print (col)
        valueCountDict={}
        colDat = mydat.ix[:,col]
        colValueCounts = pd.value_counts(colDat).sort_values(ascending=False)
        concentElement=colValueCounts.index[0]
        valueCountDict[col]=[concentElement,colValueCounts.iloc[0],colValueCounts.iloc[0]*1.0/sumCount]
        colDf = DataFrame(valueCountDict)
        colsDf = colsDf.join(colDf)
    colsDf = colsDf.rename(index={0:'concentricElement',1:'concentricCount',2:'concentricRate'}).drop('tmp',axis=1)
    return colsDf

def calcSatur(mydat,noSaturLabel=0):
    """
    :calcuate the saturability
    mydat:pd.DataFrame
    
    """
    sumCount = len(mydat.index)
    columns = mydat.columns
    colsDf = DataFrame({'tmp':['tmp',np.nan]})
    for col in columns:
        valueCountDict={}
        colDat = mydat.ix[:,col]
        saturData = colDat[colDat!=noSaturLabel].dropna()
        saturCount = len(saturData)
        saturRate = saturCount*1.0/sumCount
        valueCountDict[col]=[saturCount,saturRate]
        colDf = DataFrame(valueCountDict)
        colsDf = colsDf.join(colDf)
    colsDf = colsDf.rename(index={0:'saturCount',1:'saturRate'}).drop('tmp',axis=1)
    return colsDf
        
#if __name__=='__main__':
#    data = pd.read_csv(r'G:\getui_integ_0412.csv',encoding='gbk')
#    calcNullData = calcNull(data)
#    calcConcentricData = calcConcentric(data)
#    calcSaturData = calcSatur(data)    