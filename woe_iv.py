# -*- coding: utf-8 -*-
"""
Created on Mon May  8 23:37:30 2017

@author: evan

@target:calculate the woe and iv for the features

@note:not allow NAN

"""

import numpy as np
import pandas as pd
import math

#from sklearn import datasets
#iris = datasets.load_iris()
#raw_data = pd.DataFrame(iris.data,columns=iris.feature_names)
#raw_data['y'] =iris.target 
#ff = raw_data[raw_data['y']>0]
#ff['xx'] = ff['sepal length (cm)'].apply(lambda x:1 if x>6 else 0)
#ff['xx1'] = ff['sepal width (cm)'].apply(lambda x:1 if x>3 else 0)
#
#def vv(x):
#    if x>3 and x<3.5:
#        dd = 1
#    elif x<=3:
#        dd = 2
#    else:
#        dd=3
#    return dd
#
#ff['xx3'] = ff['sepal width (cm)'].apply(vv)

class woe_iv(object):
    def __init__(self, dataset, class_label='y', features=None):
        '''
        self._raw_data = raw data
        self._class_name = the target(y)
        self._features= features to be calculated
        
        '''

        if not isinstance(dataset, pd.core.frame.DataFrame):  # class needs a pandas dataframe
            raise AttributeError('input dataset should be a pandas data frame')
        self._raw_data = dataset
        self._class_name = class_label
        self._features = features
        
    def check_target_binary(self,y):
        #check the target is binary
        #param y:the target variable, series type
        if len(y.unique())>2:
            raise ValueError('Target must be binary!')
    
    def target_count(self,y,event=1):
        #count the 0,1 in y
        #param y:the target variable, series type
        y_count = y.value_counts()
        if event not in y_count.index:
            event_count=0
        else:
            event_count = y_count[event]
        non_event_count = len(y)-event_count
        return event_count,non_event_count
        
    def woe_iv_single_x(self,x, y, event=1,adj=0.5):
        #calculate the woe and the iv of the single feature
        #param x:the feature, series type
        #param y:the target, series type
        #return:dict of woe and the iv
        self.check_target_binary(y)
        event_count,non_event_count = self.target_count(y,event)
        feature_value_counts = x.value_counts(dropna=False)
        x_woe_dict = {}
        iv=0
        for cat in feature_value_counts.index:
            cat_event_count, cat_non_event_count = self.target_count(y[x==cat],event)
            rate_event = cat_event_count*1.0/event_count
            rate_non_event = cat_non_event_count*1.0/non_event_count
            if rate_non_event == 0:
                woe1 = math.log(((cat_event_count*1.0+adj) / event_count)/((cat_non_event_count*1.0+adj) / non_event_count))
            elif rate_event == 0:
                woe1 = math.log(((cat_event_count*1.0+adj) / event_count)/((cat_non_event_count*1.0+adj) / non_event_count))
            else:
                woe1 = math.log(rate_event / rate_non_event)
            x_woe_dict[cat]=woe1
            iv += (rate_event - rate_non_event)*woe1
        return x_woe_dict,iv
        
    def woe_iv_X(self,X,y,event=1,adj=0.5):
        #calculate the woe and the iv of the features
        #param X:the features, DataFrame type
        #param y:the target, series type
        #return:dict of woe and iv
        X_woe_dict={}
        iv_dict = {}
        for x in X.columns:
            x_woe_dict,iv = self.woe_iv_single_x(X[x],y,event,adj)
            X_woe_dict[x]=x_woe_dict
            iv_dict[x]=iv
        return X_woe_dict,iv_dict
    
    def apply_woe_iv(self):
        #calculate the woe and the iv of the features
        #return the dataframe of woe and iv
        if self._features==None:
            self._features = self._raw_data.columns.difference(pd.Index([self._class_name]))
        X = self._raw_data.loc[:,self._features]
        y = self._raw_data[self._class_name]
        X_woe_dict,iv_dict = self.woe_iv_X(X,y)
        X_woe_frame = pd.DataFrame(X_woe_dict)
        iv_frame = pd.DataFrame(pd.Series(iv_dict),columns=['iv'])
        return X_woe_frame,iv_frame
            
            
            
            
            
            
            
            
            

