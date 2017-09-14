# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:57:59 2017

@author: memedai
"""
import pandas as pd
import numpy as np
from patsy import dmatrices
from patsy import dmatrix
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


class Lr_var_stats():
    '''
    vars_vif:计算变量的vif值；
    vars_contribute：计算变量的贡献度，且倒序排列；
    vars_pvalue：计算变量的p值；
    vars_corr：计算变量的相关系数；
    all_stats：整合上面的统计指标；
    '''
    def __init__(self,mydat,lr_model,dep ='y',alpha=1):
        '''
        mydat：DataFrame,数据集，包括y；
        dep：str,y；
        lr_model:model,建好的模型；
        alpha：int，lasso中的惩罚系数，计算p值是调用statsmodels中的lasso，需重新拟合；
        vars_data：DataFrame
        '''
        self.mydat = mydat
        self.dep = dep
        self.vars_data = self.mydat[self.mydat.columns.difference([self.dep])] #特征数据集
        self.lr_model=lr_model
        self.alpha=alpha
    def vars_vif(self):
        features = "+".join(self.vars_data.columns) #生成’f1+f2+...‘ 
        X = dmatrix(features, self.vars_data, return_type='dataframe') 
        vif = pd.DataFrame()
        vif["VIF_Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])] #逐个变量计算
        vif["features"] = X.columns
        self.df_vif  = vif.set_index(['features'])
    def vars_contribute(self):
        train_x_contri = pd.DataFrame(self.vars_data.std(),columns=['std'])
        train_x_contri['coef'] = self.lr_model.coef_[0,:]
        train_x_contri['std_coef'] = 0.5513*train_x_contri['std']*train_x_contri['coef']
        train_x_contri['contribute_ratio'] = train_x_contri['std_coef']/sum(train_x_contri['std_coef'])
        train_x_contri_sort = train_x_contri.sort_values(by='contribute_ratio',ascending=False)
        self.df_contribute = pd.DataFrame(train_x_contri_sort['contribute_ratio'])
    def vars_pvalue(self):
        features = "+".join(self.vars_data.columns)          
        dep_ = self.dep+' ~'
        y, X = dmatrices(dep_ + features, self.mydat, return_type='dataframe')
        logit = sm.Logit(y, X).fit_regularized(alpha=self.alpha) #调用statsmodel中的Logit模块
        self.df_pvalue = pd.DataFrame(logit.pvalues.iloc[1:],columns=['pvalue'])
    def vars_corr(self):
        self.df_corr = pd.DataFrame(np.corrcoef(self.vars_data,rowvar=0),columns=self.vars_data.columns,index=self.vars_data.columns)
    def all_stats(self):
        self.vars_vif()
        self.vars_contribute()
        self.vars_pvalue()
        self.vars_corr()
        df_corr_trans = pd.DataFrame(self.df_corr,columns =self.df_contribute.index,index=self.df_contribute.index )#以贡献度中的变量顺序为准
        self.var_stats = self.df_contribute.merge(self.df_vif,left_index=True,right_index=True)\
                        .merge(self.df_pvalue,left_index=True,right_index=True)\
                        .merge(df_corr_trans,left_index=True,right_index=True)
