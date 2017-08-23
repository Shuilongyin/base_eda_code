# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:55:02 2017

@author: evan
"""
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Model_evaluation():
    '''
    ：roc_curve; 画roc曲线和计算auc值
    : ks_curve; 画ks曲线和计算ks值
    : group_risk_curve; 画分组图和风险曲线，并产出每个样本的分组和总的聚合分组情况
    '''
    def __init__(self,y_true,predict_prob,save_file=None):
        '''
        ：y_true; y, pd.Series, binary,(0,1)
        : predict_prob; prob, pd.Series, Continuous, (0,1)
        : save_file; file path, str, 'C:\\Users\\memedai\\Desktop\\'
        '''
        self.y = y_true #series
        self.prob = predict_prob #series
        self.prob.index = self.y.index
        self.save_file = save_file
    def __str__(self):
        return('class_Model_evaluation')
    def roc_curve(self,file_name=None):
        '''
        ：file_name; 图片名, str, 'xx.pdf'
        '''
        false_positive_rate, recall, thresholds = roc_curve(self.y,self.prob)
        roc_auc = auc(false_positive_rate, recall)        
        #fig=plt.figure()
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.ylabel('Recall')
        plt.xlabel('Fall-out')
        plt.show()
        if self.save_file is not None and file_name is not None:
            plt.savefig(self.save_file+file_name)
        else:
            pass        
        return
    def ks_curve(self,file_name=None):
        '''
        ：file_name; file name of pic, str, 'xx.pdf'
        : target: to compute the ks-value of the data
        : data: DataFrame, two columns, the first column is the real type(y), 
               the second column is the probability that y equals 1
               
        '''
        data = pd.DataFrame([self.y,self.prob]).T
        data.columns = ['y','prob']
            
        y = data.y
        
        x_axis = np.arange(len(y))/float(len(y))
        PosAll = pd.Series(y).value_counts()[1]
        NegAll = pd.Series(y).value_counts()[0]
        
        data = data.sort_values(by='prob',ascending=False)
    
        pCumsum = data['y'].cumsum()
        nCumsum = np.arange(len(y))-pCumsum+1
        pCumsumPer = pCumsum/PosAll
        nCumsumPer = nCumsum/NegAll
        
        ks = max(pCumsumPer-nCumsumPer)
        
        plt.figure(figsize=[8,6])
        plt.title('ks_curve(ks=%0.2f)'%ks)        
        plt.plot(x_axis,pCumsumPer,color='red')
        plt.plot(x_axis,nCumsumPer,color='blue')
        plt.legend(('TPR','FPR'),loc='lower right')
        plt.show()
        if self.save_file is not None and file_name is not None:
            plt.savefig(self.save_file+file_name)
        else:
            pass        
        return
    def group_risk_curve(self,n,file_name=None):
        '''
        ：n;分组的组数
        ：file_name; 图片名, str, 'xx.pdf'
        ：data,每个样本的分组详情,pd.DataFrame；
        : cut_points, 切点, 包含头尾， pd.Series
        : group_df, 分组聚合详情，pd.DataFrame
        '''
        data = pd.DataFrame([self.y,self.prob]).T
        data.columns = ['y','prob']
        prob_cuts = pd.qcut(data['prob'],q=n,labels=range(n+1)[1:],retbins=True)
        #pd.qcut(range(5), 4)  [[0, 1], [0, 1], (1, 2], (2, 3], (3, 4]] (左开右闭)     
        cuts_bin = pd.Series(prob_cuts[0])
        cut_points = pd.Series(prob_cuts[1])
        data['group'] =  cuts_bin
        data['lower_point'] = [0 for i in data.index]
        data['upper_point'] = [0 for i in data.index]
        for i in range(len(cut_points.index)-1):
            data['lower_point'][data['group']==i+1] = cut_points[i]
            data['upper_point'][data['group']==i+1] = cut_points[i+1]
        avg_risk = (data['y'].sum())/(data['y'].count())
        group = data.groupby(['group','lower_point','upper_point'],as_index=False)
        group_df = group['y'].agg({'y_count':'sum','count':'count'})
        group_df['group_per'] = group_df['count']/group_df['count'].sum()
        group_df['bad_per'] = group_df['y_count']/group_df['count']
        group_df['risk_times'] = group_df['bad_per']/avg_risk
        group_df = pd.DataFrame(group_df,columns=['group','count',\
        'y_count','group_per','bad_per','risk_times','lower_point','upper_point'])
        
        fig,ax1 = plt.subplots()  # 使用subplots()创建窗口
        ax2 = ax1.twinx() # 创建第二个坐标轴
        ax1.bar(left = group_df['group'],height = group_df['group_per'],width = 0.6,align="center",yerr=0.000001) 
        ax2.plot(list(group_df['group']),list(group_df['risk_times']),'-ro',color='red')  
        ax1.set_xlabel('group', fontsize = 12)  
        ax1.set_ylabel('percent', fontsize = 12)
        ax2.set_ylabel('risktimes', fontsize = 12)
        ax1.set_xlim([0, max(group_df['group'])+1]) 
        plt.title('group_risk_curve')
        plt.show()
        if self.save_file is not None and file_name is not None:
            plt.savefig(self.save_file+file_name)
        else:
            pass 
        
        return data,cut_points,group_df
        
class Apply_benchmark():
    '''
    ：cut_points;切点, 包含头尾， pd.Series
    : apply_benchmark_y; 测算样本带y值，在某一benchmark下的产出分组
    : apply_benchmark; 在某一benchmark下的产出分组
    '''
    def __init__(self,cut_points):
        self.cut_points = cut_points
    def apply_benchmark_y(self,y_true,predict_prob):
        '''
        ：y_true; y, pd.Series, binary,(0,1)
        : predict_prob; prob, pd.Series, Continuous, (0,1)
        : data，测算样本分组详情
        ：group_df，测算样本分组聚合详情
        '''
        predict_prob.index = y_true.index
        data = pd.DataFrame([y_true,predict_prob]).T
        data.columns = ['y','prob']        
        data['group'] = [0 for i in data.index]
        data['lower_point'] = [0 for i in data.index]
        data['upper_point'] = [0 for i in data.index]
        for i in range(len(self.cut_points)-1):
            if i==0:
                data['group'].loc[data['prob']<=self.cut_points[i+1]]=i+1
                data['lower_point'].loc[data['prob']<=self.cut_points[i+1]]=self.cut_points[i]
                data['upper_point'].loc[data['prob']<=self.cut_points[i+1]]=self.cut_points[i+1]
            elif i== len(self.cut_points)-2:
                data['group'].loc[data['prob']>self.cut_points[i]]=i+1
                data['lower_point'].loc[data['prob']>self.cut_points[i]]=self.cut_points[i]
                data['upper_point'].loc[data['prob']>self.cut_points[i]]=self.cut_points[i+1]
            else:
                data['group'].loc[(data['prob']>self.cut_points[i]) & (data['prob']<=self.cut_points[i+1])]=i+1
                data['lower_point'].loc[(data['prob']>self.cut_points[i]) & (data['prob']<=self.cut_points[i+1])]=self.cut_points[i]
                data['upper_point'].loc[(data['prob']>self.cut_points[i]) & (data['prob']<=self.cut_points[i+1])]=self.cut_points[i+1]
                
        avg_risk = (data['y'].sum())/(data['y'].count())
        group = data.groupby(['group','lower_point','upper_point'],as_index=False)
        group_df = group['y'].agg({'y_count':'sum','count':'count'})
        group_df['group_per'] = group_df['count']/group_df['count'].sum()
        group_df['bad_per'] = group_df['y_count']/group_df['count']
        group_df['risk_times'] = group_df['bad_per']/avg_risk
        group_df = pd.DataFrame(group_df,columns=['group','count',\
        'y_count','group_per','bad_per','risk_times','lower_point','upper_point'])
        return data, group_df
        
    def apply_benchmark(self,predict_prob):
        '''
        : predict_prob; prob, pd.Series, Continuous, (0,1)
        : data，测算样本分组详情,pd.DataFrame
        ：group_df，测算样本分组聚合详情,pd.DataFrame
        '''
        data = pd.DataFrame(predict_prob,columns=['prob'])
        data['group'] = [0 for i in data.index]
        data['lower_point'] = [0 for i in data.index]
        data['upper_point'] = [0 for i in data.index]
        for i in range(len(self.cut_points)-1):
            if i==0:
                data['group'].loc[data['prob']<=self.cut_points[i+1]]=i+1
                data['lower_point'].loc[data['prob']<=self.cut_points[i+1]]=self.cut_points[i]
                data['upper_point'].loc[data['prob']<=self.cut_points[i+1]]=self.cut_points[i+1]
            elif i== len(self.cut_points)-2:
                data['group'].loc[data['prob']>self.cut_points[i]]=i+1
                data['lower_point'].loc[data['prob']>self.cut_points[i]]=self.cut_points[i]
                data['upper_point'].loc[data['prob']>self.cut_points[i]]=self.cut_points[i+1]
            else:
                data['group'].loc[(data['prob']>self.cut_points[i]) & (data['prob']<=self.cut_points[i+1])]=i+1
                data['lower_point'].loc[(data['prob']>self.cut_points[i]) & (data['prob']<=self.cut_points[i+1])]=self.cut_points[i]
                data['upper_point'].loc[(data['prob']>self.cut_points[i]) & (data['prob']<=self.cut_points[i+1])]=self.cut_points[i+1]
        group = data.groupby(['group','lower_point','upper_point'],as_index=False)
        group_df = group['prob'].agg({'count':'count'})
        group_df['group_per'] = group_df['count']/group_df['count'].sum()
        group_df = pd.DataFrame(group_df,columns=['group','count',\
        'group_per','lower_point','upper_point'])
        return data,group_df











