# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:37:27 2017

@author: evan
"""

import numpy as np
import pandas as pd
import Orange
import copy
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable


class Df2table():
    '''
    series2descriptor:判断转成orange_table列的形式，是连续值，还是离散值；
    df2domain：生成orange_table的domain内容，分有无因变量；domain存储列的名称、类型等，其本身有很多特性，如attributes;
    Df2table:将DataFrame转成orange中的table格式    
    '''
    def __init__(self, df=None, dep='y', no_dep=False):
        '''
        self.df:DataFrame,待转换的DataFrame；
        self.dep:str,因变量的列名；
        self.no_dep:boolean,判断DataFrame中是否有因变量，默认False
        self.domain:domain,orange中table的domain；
        self.table:orange.table,转变后的table;
        '''
        self.df = df
        self.dep = dep
        self.no_dep = no_dep
        self.domain = None
        self.table = None
        self.df2domain()
        self.Df2table()
    def series2descriptor(self,d):
        '''
        d:Series
        '''
        if 'float' in str(d.dtype) or 'int' in str(d.dtype):
            return ContinuousVariable(str(d.name)) #如果dtype是int或float格式，则认为是连续型
        else:
            t = d.unique()
            t.sort()
            return DiscreteVariable(str(d.name), list(t.astype("str"))) #否则是离散型
    def df2domain(self):
        if self.no_dep: #如果没有因变量
            #生成每列的domain信息
            featurelist = [self.series2descriptor(self.df.iloc[:,col]) for col in range(len(self.df.columns))]
            self.domain = Domain(featurelist)
        else:#如果有因变量
            df_fea = self.df.loc[:,self.df.columns.difference(pd.Index([self.dep]))] #先把因变量从列中剔除
            #自变量的domain-list
            featurelist = [self.series2descriptor(df_fea.iloc[:,col]) for col in range(len(df_fea.columns))]
            #因变量的domain
            dep_domain = self.series2descriptor(self.df[self.dep])
            #合并二者
            self.domain = Domain(featurelist, dep_domain)
    def Df2table(self):
        if self.no_dep: #如果没有因变量
            self.table = Table(self.domain, np.array(self.df))
        else: #如果有因变量，需将因变量的列数据单独作为一项输入
            df_fea = self.df.loc[:,self.df.columns.difference(pd.Index([self.dep]))]
            self.table = Table(self.domain, np.array(df_fea),np.array(self.df[self.dep]))


#若用mdlp，则y必须是DiscreteVariable，所以注意将输入的df['y']的type转成object
#区间是左闭右开            

class Orange_dis():
    '''
    orange_cut_points:产生离散切点，现在有三种离散方法,等值,等频和MDLP，等频有时并不是绝对意义上的每段（量）相等，
        比如当有一列出现大量重复值是，即num/sum>1/n，所以这样会在重复值以外，再均分n-1组，这
        n-1组的数量就会比第一组要少。
        MDLP有两个需要注意的地方，df转成orange table之前，因变量需转成str，否则转成orange table
        之后再离散会报错；另外一个是MDLP有离散阈值，当某一特征所有的切点都达不到这个阈值时，不会产
        生离散切点，如果要强制离散，需要将force设置True(现默认True)；
    var_apply_cutpoints：单变量根据切点list进行离散，离散的组数从0开始；
    vars_apply_cutpoints：多变量离散；
    vars_cutpoints_group：计算多个变量每组的占比和逾期比例，只有当有因变量时才可以，且响应值=1；
    var_cutpoint_adjust：对单变量离散切点进行调整，调整的依据是某组数量不能太小，这里的调整输入数据集中需要因变量；
    vars_cutpoint_adjust:对多变量离散切点进行调整
    orange_dis：汇总方法
    '''
    def __init__(self, df=None, cut_points=None, method='mdlp', n=4, nodiscol=None, bins_threshhold=10\
                 ,dep='y', no_dep=False, per_threshhold=0.03):
        '''
        self.df：DataFrame，需要dis的特征数据集和因变量，不需要dis的特征最好不要放进来；
        self.cut_points：dict，存储变量切点,若是输入项，则变量不进行离散；
        self.cut_points_adjust：dict,存储调整后的切点；
        self.method：str，离散方法；
        self.n：int,离散组数;
        self.dep:str,因变量的列名；
        self.no_dep:boolean，是否有应变量，默认False,设为True时，记得调整离散方法
        self.nodiscol:list，不需要离散的变量名，若未提前给定，系统自动判断
        self.bins_threshold:int,若未给定不需离散的变量名称，则看变量中不同数值的个数，判断是否需要离散；
        self.per_threshhold:float,组中样本数占比阈值；
        self.df_dis_bins：DataFrame,切点未调整前，离散后的数据集；
        self.df_dis_bins_:DataFrame,切点未调整前，离散后的数据集(更易读)；
        self.df_dis_bins_adjust：DataFrame,切点调整后，离散后的数据集；
        self.df_dis_bins_adjust_:DataFrame,切点调整后，离散后的数据集(更易读)；
        self.vars_group_dict:dict,切点未调整前，每个特征每组的汇总指标；
        self.vars_group_dict_:dict,切点未调整前，每个特征每组的汇总指标(更易读)；
        self.vars_group_dict_adjust:dict,切点调整后，每个特征每组的汇总指标；
        self.vars_group_dict_adjust_:dict,切点调整后，每个特征每组的汇总指标(更易读)；
        '''
        self.df = df
        self.cut_points = cut_points
        self.cut_points_adjust = None
        self.method = method
        self.n = n
        self.dep = dep
        self.no_dep = no_dep
        self.nodiscol = nodiscol
        self.bins_threshhold = bins_threshhold
        self.per_threshhold= per_threshhold
        self.df_dis_bins = None
        self.df_dis_bins_ = None
        self.df_dis_bins_adjust = None
        self.df_dis_bins_adjust_ = None
        self.vars_group_dict = None   
        self.vars_group_dict_ = None 
        self.vars_group_dict_adjust = None  
        self.vars_group_dict_adjust_ = None  
        if cut_points==None and nodiscol==None: #若cut_points未给定，nodiscol未给定
            self.nodiscol=[]
            for var in df.columns.difference(pd.Index([self.dep])):#遍历特征
                if len(df[var].unique())<self.bins_threshhold: #若特征不同数值个数小于给定阈值
                    self.nodiscol.append(var) #添加
        self.orange_dis()
    def orange_cut_points(self, df, method='mdlp', n=4):
        '''
        df:DataFrame,输入的特征集；
        method:str，离散方法，目前width-等值，freq-等频，mdlp-MDLP；
        n:int，离散组数，仅针对等值和等频离散；
        
        ：vars_cut_points，dict,最后产出各个变量的切点从小到大排序（左闭右开）
        '''
        vars_cut_points = {}
        if self.dep in df.columns: #判断有没有因变量
            df[self.dep] = df[self.dep].astype(np.str) #若有，则将数据格式转成str，适用于mdlp的orange-table
        orange_table = Df2table(df, self.dep, self.no_dep).table #转成orange-table
        if method == 'width':
            disc = Orange.preprocess.Discretize()
            disc.method = Orange.preprocess.discretize.EqualWidth(n)
        elif method == 'freq':
            disc = Orange.preprocess.Discretize()
            disc.method = Orange.preprocess.discretize.EqualFreq(n)
        else:
            disc = Orange.preprocess.Discretize()
            disc.method = Orange.preprocess.discretize.EntropyMDL(force=True)            
        d_disc_table = disc(orange_table) #离散，得到离散后的orange-table
        #离散的明细藏在domain里面，我们只要取出每个特征的切点即可
        for i in range(len(d_disc_table.domain)-1):
            vars_cut_points[d_disc_table.domain[i].name] = [i for i in d_disc_table.domain[i].compute_value.points]
        return vars_cut_points#从小到大排序
    def var_apply_cutpoints(self, var_series, var_cut_points):
        '''
        var_series:Series,变量的Series值；
        var_cut_points：list，变量的切点；
        
        ：var_series_copy，Series，即离散后的变量，组数从0开始，缺失值是-1；
        ：var_series_copy_，Series,即离散后的变量，组的标志更易读；
        '''
        if len(var_cut_points)==0: #如果没切点，不离散
            return var_series, var_series
        else:
            var_series_copy = copy.deepcopy(var_series)
            var_series_copy_ = copy.deepcopy(var_series)
            if len(var_cut_points) == 1: #如果只有一个切点
                var_series_copy[var_series<var_cut_points[0]] = 0
                var_series_copy[var_series>=var_cut_points[0]] = 1
                var_series_copy[pd.isnull(var_series)] = -1
                var_series_copy_[var_series<var_cut_points[0]] = '<%s'%var_cut_points[0]
                var_series_copy_[var_series>=var_cut_points[0]] = '>=%s'%var_cut_points[0]
                var_series_copy_[pd.isnull(var_series)] == 'NAN'
            else:
                var_series_copy[pd.isnull(var_series)] = -1
                var_series_copy_[pd.isnull(var_series)] = 'NAN'
                for i in range(len(var_cut_points)+1):
                    if i == 0: #如果是最小的切点
                        var_series_copy[var_series<var_cut_points[0]] = 0
                        var_series_copy_[var_series<var_cut_points[0]] = '<%s'%var_cut_points[0]
                    elif i == len(var_cut_points): #如果是最大的切点
                        var_series_copy[var_series>=var_cut_points[i-1]] = i
                        var_series_copy_[var_series>=var_cut_points[i-1]] = '>=%s'%var_cut_points[i-1]
                    else:
                        var_series_copy[(var_series>=var_cut_points[i-1])&(var_series<var_cut_points[i])] = i
                        var_series_copy_[(var_series>=var_cut_points[i-1])&(var_series<var_cut_points[i])] = '[%s,%s)'%(var_cut_points[i-1],var_cut_points[i])
        return var_series_copy, var_series_copy_
    def vars_apply_cutpoints(self, df, vars_cut_points):
        '''
        df:DataFrame,待离散的的变量数据集；
        vars_cut_points：dict,变量集的切点；
        
        :df_copy：DataFrame,如var_series_copy；
        d:f_copy_：DataFrame,如var_series_copy_；
        '''
        df_copy = copy.deepcopy(df)
        df_copy_ = copy.deepcopy(df)
        for var_name in vars_cut_points.keys():
            if var_name not in df.columns:
                pass
            else:
                var_series_copy, var_series_copy_ = self.var_apply_cutpoints(df[var_name], vars_cut_points[var_name])
                df_copy[var_name] = var_series_copy
                df_copy_[var_name] = var_series_copy_
        return df_copy, df_copy_
    def vars_cutpoints_group(self, df_bins, group_na=False, fillna = -1):
        '''
        df_bins:DataFrame,包含因变量的离散特征数据集；
        group_na：Boolean，group时，是否需要计算na的指标；
        fillna：str or int，填充na的值；
        
        ：vars_group_dict，dict，包含每个特征的group指标；
        '''
        vars_group_dict = {}
        vars_name = df_bins.columns.difference(pd.Index([self.dep]))
        df_bins['y'] = df_bins['y'].astype(np.float64)
        samples_num = len(df_bins.index)
        avg_risk = sum(df_bins['y'])/samples_num
        if group_na:
            df_bins_copy = copy.deepcopy(df_bins)
            df_bins_copy.fillna(fillna, inplace=True)
        else:
            df_bins_copy = copy.deepcopy(df_bins.replace({-1: np.nan}))
        for var in vars_name:
            var_group = df_bins_copy.groupby(by=var, as_index=False)[self.dep].agg({'bad_count':'sum','count':'count'})
            var_group['per'] = var_group['count']/samples_num
            var_group['bad_per'] = var_group['bad_count']/var_group['count']
            var_group['risk_times'] = var_group['bad_per']/avg_risk
            vars_group_dict[var] = var_group
        return vars_group_dict
    def var_cutpoint_adjust(self, var_name, per_threshhold=0.03):
        '''
        var_name:str,需要调整切点的变量；
        per_threshhold：float，某组样本数占比的阈值；
        '''
        var_cutpoints = {} #
        var_cutpoints[var_name] = self.cut_points_adjust[var_name] #将变量切点取出
        if len(var_cutpoints[var_name]) <=1: #如果只有一个切点，不调整
            pass
        else:
            #计算每组指标
            var_bins_y, var_bins_y_ = self.vars_apply_cutpoints(self.df.loc[:,[var_name,self.dep]], var_cutpoints)
            var_group_dict = self.vars_cutpoints_group(var_bins_y)
            var_group = var_group_dict[var_name]
            var_bins_lower = var_group[var_group['per']<per_threshhold][var_name].tolist() #取小于阈值的组，并将组存于list中
            var_bins = var_group[var_name].tolist() #将变量的组存于list中，用于后续比较
            if len(var_bins_lower)==0: #如果没有小于阈值的组，则不调整
                pass
            else:
                var_bin_lower_last = var_bins_lower[-1] #取不满足阈值的最大的组
                if var_bin_lower_last == max(var_bins) or var_bin_lower_last == min(var_bins): #如果该组在首或尾，则直接合并到邻组
                    del self.cut_points_adjust[var_name][int(var_bin_lower_last-1)]
                else:#否则，比较逾期率与哪个组相近，则合并到相似的组
                    bad_per_diff_left = abs(var_group[var_group[var_name]==var_bin_lower_last]['bad_per'].values[0]-\
                                               var_group[var_group[var_name]==(var_bin_lower_last-1)]['bad_per'].values[0])
                    bad_per_diff_right = abs(var_group[var_group[var_name]==var_bin_lower_last]['bad_per'].values[0]-\
                                               var_group[var_group[var_name]==(var_bin_lower_last+1)]['bad_per'].values[0])
                    if bad_per_diff_left<bad_per_diff_right:
                        del self.cut_points_adjust[var_name][int(var_bin_lower_last-1)]
                    else:
                        del self.cut_points_adjust[var_name][int(var_bin_lower_last)]
                self.var_cutpoint_adjust(var_name, per_threshhold) #递归
        return
    def vars_cutpoint_adjust(self, per_threshhold=0.03):
        for var in self.cut_points.keys():
            self.var_cutpoint_adjust(var, per_threshhold)
        return 
    def orange_dis(self):
        if self.nodiscol is not None:
            discol_y = self.df.columns.difference(pd.Index(self.nodiscol)) #剔除不需要离散的特征名
            discol_y_data = self.df.loc[:,discol_y] #取需要离散的数据集
        else:
            discol_y_data = self.df
        if self.no_dep:#若没有y值，则method必须调整，且没有调整切点的步骤，也没有计算每组逾期率的步骤
            if self.cut_points is None: #若切点未给定
                self.cut_points = self.orange_cut_points(discol_y_data, self.method, self.n) #找切点
                self.df_dis_bins, self.df_dis_bins_ = self.vars_apply_cutpoints(self.df, self.cut_points) #根据切点离散
            else: #若切点已给定，直接根据已知切点离散
                self.df_dis_bins, self.df_dis_bins_ = self.vars_apply_cutpoints(self.df, self.cut_points) #根据切点离散
        else: #如有y值    
            if self.cut_points is None: #若切点未给定
                self.cut_points = self.orange_cut_points(discol_y_data, self.method, self.n) #寻找切点
                self.df_dis_bins, self.df_dis_bins_ = self.vars_apply_cutpoints(self.df, self.cut_points) #根据切点离散
                self.vars_group_dict = self.vars_cutpoints_group(self.df_dis_bins,group_na=True) #计算变量每组的情况
                self.vars_group_dict_ = self.vars_cutpoints_group(self.df_dis_bins_,group_na=True,fillna='NAN') #计算变量每组的情况（易读）
                
                self.cut_points_adjust = copy.deepcopy(self.cut_points) 
                self.vars_cutpoint_adjust(self.per_threshhold) #调整切点
                self.df_dis_bins_adjust, self.df_dis_bins_adjust_ = self.vars_apply_cutpoints(self.df, self.cut_points_adjust) #根据调整的切点离散
                self.vars_group_dict_adjust = self.vars_cutpoints_group(self.df_dis_bins_adjust,group_na=True) #切点调整后变量每组情况
                self.vars_group_dict_adjust_ = self.vars_cutpoints_group(self.df_dis_bins_adjust_,group_na=True,fillna='NAN') #切点调整后变量每组情况（易读）
            else:
                self.df_dis_bins, self.df_dis_bins_ = self.vars_apply_cutpoints(self.df, self.cut_points) #根据给定切点离散
                self.vars_group_dict = self.vars_cutpoints_group(self.df_dis_bins,group_na=True) #计算变量每组的情况
                self.vars_group_dict_ = self.vars_cutpoints_group(self.df_dis_bins_,group_na=True,fillna='NAN') #计算变量每组的情况（易读）            
        return 
        
    
            


































