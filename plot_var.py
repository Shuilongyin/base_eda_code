# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:05:12 2017

@author: evan
"""
import pandas as pd
pd.set_option('precision', 4)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 
matplotlib.rcParams['font.serif'] = ['SimHei'] 
font = { "family":"SimHei"}
matplotlib.rc("font",**font)

class Plot_vars():
    '''
    cut_points_bring:单变量，针对连续变量离散化，目标是均分xx组，但当某一数值重复过高时会适当调整，最后产出的是分割点，不包含首尾
    dis_group：单变量，离散化连续变量，并生成一个groupby数据；
    nodis_group：不需要离散化的变量，生成一个groupby数据；
    plot_var：单变量绘图
    plot_vars：多变量绘图
    '''
    def __init__(self,mydat,dep='y',nodiscol=None,plotcol=None,disnums=5,file_path=None):
        '''
        mydat:DataFrame,包含X,y的数据集；
        dep：str,the label of y；
        plotcol:list,defalut None,当这个变量有数据时，调用多变量绘图只会绘画该list中的变量；
        nodiscol：list,defalut None，当这个变量有数据时，会默认这里的变量不离散，且只画nodis_group，；
            其余的变量都需要离散化，且只画dis_group。当这个变量为空时，系统回去计算各变量的不同数值的数量，若小于15
            ，则认为不需要离散，直接丢到 nodiscol中；
        disnums：int，连续变量需要离散的组数；
        file_path：保存图像的文件路径，如'C:\\Users\\memedai\\Desktop\\'；
        col_cut_points：用来保存连续变量的分割点；
        col_notnull_count：保存变量非null的数量，用于离散化计算分割比例；
        '''
        self.mydat = mydat #输入数据集
        self.dep = dep #y的标签
        self.plotcol = plotcol #需要画图的变量
        self.nodiscol = nodiscol #不需要离散的变量
        self.disnums = disnums #分割组数
        self.file_path=file_path #
        self.col_cut_points={}
        self.col_notnull_count={}
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
        col_group = dis_col.groupby([col],as_index=False)['y'].agg({'count':'count','bad_num':'sum'})
        avg_risk = self.mydat[self.dep].sum()/self.mydat[self.dep].count() #平均风险
        col_group['per'] = col_group['count']/col_group['count'].sum()
        col_group['bad_per'] = col_group['bad_num']/col_group['count']
        col_group['risk_times'] = col_group['bad_per']/avg_risk
        if -1 in list(col_group[col]):#判断是否有缺失值，关系到列的长度
            col_group['bins'] = dis_col_bins
        else:
            col_group['bins'] = dis_col_bins[1:]
        col_group[col] = col_group[col].astype(np.float)
        col_group = col_group.sort_values([col],ascending=True)#再一次排序
        col_group = pd.DataFrame(col_group,columns=[col,'bins','per','bad_per','risk_times'])
        return col_group
    def nodis_group(self,col):
        '''
        col:str,变量名称
        '''
        nodis_col_data = self.mydat.loc[:,[self.dep,col]] #取该变量的数据，包括null值
        is_na = (pd.isnull(nodis_col_data[col]).sum()>0) #判断是否有null值
        col_group = nodis_col_data.groupby([col],as_index=False)['y'].agg({'count':'count','bad_num':'sum'})
        col_group = pd.DataFrame(col_group,columns=[col,'bad_num','count'])    
        if is_na:#判断是否有null值，若有，则单独计算，因null值不存在上面的col_group中
            y_na_count = nodis_col_data[pd.isnull(nodis_col_data[col])][self.dep].count()
            y_na_sum = nodis_col_data[pd.isnull(nodis_col_data[col])][self.dep].sum()
            col_group.loc[len(col_group.index),:] = [-1,y_na_sum,y_na_count]#添加一行
        avg_risk = self.mydat[self.dep].sum()/self.mydat[self.dep].count() #平均风险
        col_group['per'] = col_group['count']/col_group['count'].sum()
        col_group['bad_per'] = col_group['bad_num']/col_group['count']
        col_group['risk_times'] = col_group['bad_per']/avg_risk
        if is_na: #判断是否有null值，增加一列bins
            bins = col_group[col][:len(col_group.index)-1]
            bins[len(bins.index)]='nan'
            col_group['bins'] = bins
            col_labels = list(range(len(col_group.index)-1))
            col_labels.append(-1)
            col_group[col] = col_labels
        else:
            bins = col_group[col]
            col_group['bins'] = bins
            col_labels = list(range(len(col_group.index)))
            col_group[col] = col_labels
        col_group[col] = col_group[col].astype(np.float)
        col_group = col_group.sort_values([col],ascending=True)#排序
        col_group = pd.DataFrame(col_group,columns=[col,'bins','per','bad_per','risk_times'])
        return col_group
    def plot_var(self,col):
        '''
        col:str,变量名称
        '''
        if self.nodiscol is not None and col in self.nodiscol: #判断col需不需要离散化
            col_group = self.nodis_group(col)
        else: 
            col_group = self.dis_group(col)
        fig,ax1 = plt.subplots()  # 使用subplots()创建窗口
        #先画柱形图
        ax1.bar(left = list(col_group[col]),height = col_group['per'],width = 0.6,align="center") 
        #添加数据标签
        for a,b in zip(col_group[col],col_group['per']):
            ax1.text(a, b+0.005, '%.4f' % b, ha='center', va= 'bottom',fontsize=7)        
        ax1.set_xlabel(col)  
        ax1.set_ylabel('percent', fontsize = 12)   
        ax1.set_xlim([-2, max(col_group[col])+2]) 
        ax1.set_ylim([0, max(col_group['per'])+0.3])
        ax1.grid(False) #不要网格线       

        ax2 = ax1.twinx() # 创建第二个坐标轴
        ax2.plot(list(col_group[col]),list(col_group['risk_times']),'-ro',color='red')#折线图
        #添加数据标签                   
        for a,b in zip(col_group[col],col_group['risk_times']):
            ax2.text(a, b+0.05, '%.4f' % b, ha='center', va= 'bottom',fontsize=7) 
        ax2.set_ylabel('risktimes', fontsize = 12)
        ax2.set_ylim([0, max(col_group['risk_times'])+0.5])
        ax2.grid(False)
        plt.title(col)
        #将数据详情表添加
        the_table = plt.table(cellText=col_group.round(4).values,\
                          colWidths = [0.1]*len(col_group.columns),\
                          rowLabels=col_group.index,\
                          colLabels=col_group.columns,\
                          loc=1)
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(5) #设置表格的字大小
        if self.file_path is not None:#判断是否需要保存
            plt.savefig(self.file_path+col+'.pdf',dpi=600)
        plt.show()
        return
    def plot_vars(self):
        if self.plotcol is not None: #若plotcol有指定，则只画指定的变量
            for col in self.plotcol:
                self.plot_var(col)
        else:
            cols = self.mydat.columns #若没有指定，则全部画
            for col in cols:
                print(col)
                if col!=self.dep:
                    self.plot_var(col)
        return