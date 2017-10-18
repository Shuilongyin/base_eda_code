# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:16:47 2017

@author: evan
"""
import numpy as np
import math
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from copy import copy


class Woe_iv():
    '''
    check_target_binary:检查是否是二分类问题；
    target_count：计算好坏样本数目；
    cut_points_bring：单变量，针对连续变量离散化，目标是均分xx组，但当某一数值重复过高时会适当调整，最后产出的是分割点，不包含首尾
    dis_group：单变量，离散化连续变量，并生成一个groupby数据；
    nodis_group：不需要离散化的变量，生成一个groupby数据；
    woe_iv_var：单变量，计算各段的woe值和iv
    woe_iv_vars：多变量，计算多变量的woe和iv
    apply_woe_replace：将数据集中的分段替换成对应的woe值
    '''
    def __init__(self,mydat,dep='y',event=1,nodiscol=None,ivcol=None,disnums=20,X_woe_dict=None):
        '''
        mydat：DataFrame,输入的数据集，包括y；
        event：int,y中bad的标签；
        nodiscol：list,defalut None，不需要离散的变量名，当这个变量有数据时，会默认这里的变量不离散，且只跑nodis_group，；
            其余的变量都需要离散化，且只跑dis_group。当这个变量为空时，系统会去计算各变量的不同数值的数量，若小于15
            ，则认为不需要离散，直接丢到 nodiscol中；
        ivcol：list,需要计算woe，iv的变量名，该变量不为None时，只跑这些变量，否则跑全体变量；
        disnums：int，连续变量离散化的组数；
        X_woe_dict：dict，每个变量每段的woe值，这个变量主要是为了将数据集中的分段替换成对应的woe值，
            即输入的数据集已经经过离散化分段处理，只需要woe化而已。
        '''
        self.mydat = mydat
        self.event = event
        self.nodiscol = nodiscol
        self.ivcol = ivcol
        self.disnums=disnums
        self._WOE_MIN = -20 #暂时没用
        self._WOE_MAX = 20 #暂时没用
        self.dep = dep
        self.col_cut_points={} #为各个变量生成一个空list，用于后面保存分割点，
                                            #由于分割是一个递归，所以返回文件需要提前定义
        self.col_notnull_count={}   #变量非null的数量
        self._data_new = self.mydat.copy(deep = True) #copy一个新的数据集，用来存储分组和woe化
        self.X_woe_dict = X_woe_dict 
        self.iv_dict=None #用来存储变量iv
        for i in self.mydat.columns:
            if i != self.dep:
                self.col_cut_points[i]=[] #为各个变量生成一个空list，用于后面保存分割点，
                                            #由于分割是一个递归，所以返回文件需要提前定义
        for i in self.mydat.columns:
            if i != self.dep :
                col_notnull = len(self.mydat[i][pd.notnull(self.mydat[i])].index)#变量非null的数量
                self.col_notnull_count[i]= col_notnull
        if self.nodiscol is None: #如果人为没有定义哪些变量需要离散，则系统自动筛选
            nodiscol_tmp=[]
            for i in self.mydat.columns:
                if i != self.dep:
                    col_cat_num = len(set(self.mydat[i][pd.notnull(self.mydat[i])]))
                    if col_cat_num<15:
                        nodiscol_tmp.append(i)
            if len(nodiscol_tmp)>0:
                self.nodiscol = nodiscol_tmp        

    def check_target_binary(self, y):
        '''
        check if the target variable is binary, raise error if not.
        :param y:
        :return:
        '''
        y_type = type_of_target(y)
        if y_type not in ['binary']:
            raise ValueError('Label type must be binary')

    def target_count(self,y):
        '''
        #count the 0,1 in y
        #param y:the target variable, series type
        '''
        y_count = y.value_counts()
        if self.event not in y_count.index:
            event_count=0
        else:
            event_count = y_count[self.event]
        non_event_count = len(y)-event_count
        return event_count,non_event_count
        
    def cut_points_bring(self,col_order,col):
        '''
        col_order:DataFrame,非null的数据集，包含y，按变量值顺序排列；
        col:str.变量名；
        disnums：离散的组数；disnums每找到一个分割点，就减1
        '''
        PCount = len(col_order.index)
        min_group_num = self.col_notnull_count[col]/self.disnums
        disnums = int(PCount/min_group_num)  #重新根据剩下的数据确定分割组数
        if PCount/self.col_notnull_count[col]>=(1/self.disnums) and disnums>0:#若剩下的数据满足分组条件，即剩下的数据的占比不小于xx比例，则继续分
            n_cut = int(PCount/disnums) #找到分割点
            cut_point = col_order[col].iloc[n_cut-1] #分割点的数值
            for i in col_order[col].iloc[n_cut:]: #看这个数值在这个点是否唯一，若不唯一，往后推
                if i==cut_point:
                    n_cut+=1
                else:
                    self.col_cut_points[col].append(cut_point)
                    break
            self.cut_points_bring(col_order[n_cut:],col) #递归，在剩下的数据集中再分割

    def dis_group(self,col):
        '''
        col:str,变量名称
        '''
        dis_col_data_notnull = self.mydat.loc[pd.notnull(self.mydat[col]),[self.dep,col]] #取出非null数据，包括y        
        Order_P=dis_col_data_notnull.sort_values(by=[col],ascending=True)#排序
        self.cut_points_bring(Order_P,col) #取得分割点
        dis_col_cuts=[]
        dis_col_cuts.append(dis_col_data_notnull[col].min())
        dis_col_cuts.extend(self.col_cut_points[col])
        dis_col_cuts.append(dis_col_data_notnull[col].max())#将首尾拿进来
        dis_col_data = self.mydat.loc[:,[self.dep,col]] #取该变量全数据，包括null值
        dis_col_data['group']=np.nan #生成一个分组列，赋值null
        for i in range(len(dis_col_cuts)-1):#根据分割点分组
            if i==0:
                dis_col_data['group'].loc[dis_col_data[col]<=dis_col_cuts[i+1]]=i
            elif i== len(dis_col_cuts)-2:
                dis_col_data['group'].loc[dis_col_data[col]>dis_col_cuts[i]]=i
            else:
                dis_col_data['group'].loc[(dis_col_data[col]>dis_col_cuts[i]) & (dis_col_data[col]<=dis_col_cuts[i+1])]=i       
        dis_col_data[col] = dis_col_data['group'] #将分组赋值到原变量值列
        dis_col_bins=[]  
        dis_col_bins.append('nan') #将null值放bins的首位（因-1排第一）
        #生成一个左开右闭的字符串范围
        dis_col_bins.extend( ['(%s,%s]'%(dis_col_cuts[i],dis_col_cuts[i+1]) for i in range(len(dis_col_cuts)-1)])
        dis_col = dis_col_data.fillna(-1) #将nan值赋值为-1
        col_group = dis_col.groupby([col],as_index=False)[self.dep].agg({'count':'count','bad_num':'sum'})
        col_group['good_num'] = col_group['count']-col_group['bad_num']
        col_group['per'] = col_group['count']/col_group['count'].sum()
        if -1 in list(col_group[col]):#判断是否有缺失值，关系到列的长度
            col_group['bins'] = dis_col_bins
        else:
            col_group['bins'] = dis_col_bins[1:]
        for i in range(len(dis_col_cuts)-1):#用_data_new来存储分组，分组的标签用的区间字符串
            if i==0:
                 self._data_new[col].loc[self.mydat[col]<=dis_col_cuts[i+1]]=dis_col_bins[i+1]
            elif i== len(dis_col_cuts)-2:
                self._data_new[col].loc[self.mydat[col]>dis_col_cuts[i]]=dis_col_bins[i+1]
            else:
                self._data_new[col].loc[(self.mydat[col]>dis_col_cuts[i]) & (self.mydat[col]<=dis_col_cuts[i+1])]=dis_col_bins[i+1]       
        self._data_new.fillna(value='nan',inplace=True) #缺失值用'nan'字符串填充，这里应该与woe_dict中的key的标签一致
        return col_group
    def nodis_group(self,col):
        '''
        col:str,变量名称
        '''
        nodis_col_data = self.mydat.loc[:,[self.dep,col]] #取该变量的数据，包括null值
        is_na = (pd.isnull(nodis_col_data[col]).sum()>0) #判断是否有null值
        col_group = nodis_col_data.groupby([col],as_index=False)[self.dep].agg({'count':'count','bad_num':'sum'})
        col_group = pd.DataFrame(col_group,columns=[col,'bad_num','count'])    
        if is_na:#判断是否有null值，若有，则单独计算，因null值不存在上面的col_group中
            y_na_count = nodis_col_data[pd.isnull(nodis_col_data[col])][self.dep].count()
            y_na_sum = nodis_col_data[pd.isnull(nodis_col_data[col])][self.dep].sum()
            col_group.loc[len(col_group.index),:] = [-1,y_na_sum,y_na_count]#添加一行
        col_group['good_num'] = col_group['count']-col_group['bad_num']
        col_group['per'] = col_group['count']/col_group['count'].sum()
        if is_na: #判断是否有null值，增加一列bins
            bins = col_group[col][:len(col_group.index)-1]
            bins.loc[len(bins.index)]='nan'
            col_group['bins'] = bins
            col_labels = list(range(len(col_group.index)-1))
            col_labels.append(-1)
            col_group[col] = col_labels
        else:
            bins = col_group[col]
            col_group['bins'] = bins
            col_labels = list(range(len(col_group.index)))
            col_group[col] = col_labels
        return col_group

    def woe_iv_var(self,col, adj=0.5):
        '''
        col:str,变量名称
        adj:float,分母为0时的异常调整
        '''
        self.check_target_binary(self.mydat[self.dep])
        event_count,non_event_count = self.target_count(self.mydat[self.dep])
        if self.nodiscol is not None and col in  self.nodiscol: #判断col跑哪一种group方法
            col_group = self.nodis_group(col)
        else:
            col_group = self.dis_group(col)
        x_woe_dict = {}
        iv=0
        for cat in col_group['bins']:#对每一分段循环计算woe和iv
            cat_event_count = col_group.loc[col_group.loc[:,'bins']==cat,'bad_num'].iloc[0]
            cat_non_event_count = col_group.loc[col_group.loc[:,'bins']==cat,'good_num'].iloc[0]
            rate_event = cat_event_count*1.0/event_count
            rate_non_event = cat_non_event_count*1.0/non_event_count
            if rate_non_event == 0: #若cat_non_event_count为0
                woe1 = math.log(((cat_event_count*1.0+adj) / event_count)/((cat_non_event_count*1.0+adj) / non_event_count))
            elif rate_event == 0: #若cat_event_count为0
                woe1 = math.log(((cat_event_count*1.0+adj) / event_count)/((cat_non_event_count*1.0+adj) / non_event_count))
            else:
                woe1 = math.log(rate_event / rate_non_event)
            x_woe_dict[cat]=woe1 #每段woe存储
            iv += (rate_event - rate_non_event)*woe1  #iv累积          
        return x_woe_dict,iv
    def woe_iv_vars(self,adj=0.5):
        X_woe_dict={}
        iv_dict = {}
        if self.ivcol is not None: #判断ivcol是否为None值，若不为None，则只跑这些变量
            for col in self.ivcol:
                print (col)
                x_woe_dict,iv = self.woe_iv_var(col,adj)
                X_woe_dict[col]=x_woe_dict
                iv_dict[col]=iv
        else:
            for col in self.mydat.columns:
                print(col)
                if col!=self.dep:
                    x_woe_dict,iv = self.woe_iv_var(col,adj)
                    X_woe_dict[col]=x_woe_dict
                    iv_dict[col]=iv
        self.X_woe_dict= X_woe_dict #将计算得到的变量woe值赋值给class的X_woe_dict的属性
        self.iv_dict = iv_dict
    def apply_woe_replace(self):
        #将每一变量的每一段进行woe替换
        for col in self.X_woe_dict.keys():
            for binn in self.X_woe_dict[col].keys():
                self._data_new.loc[self._data_new.loc[:,col]==binn,col]=self.X_woe_dict[col][binn]
        self._data_new = self._data_new.astype(np.float64)
        return









