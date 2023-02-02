# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 23:57:52 2023

@author: Hangyu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
from tqdm import trange
import pickle
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from KalmanFilter import *       
        
class fund_position_measure:
    
    def __init__(self):
        #导入基金持仓数据
        with open('./data/fund_position_total.pickle','rb') as f:
            fund_position_total = pickle.load(f)
        #导入基金净值数据
        with open('./data/fund_nav_total.pickle','rb') as f:
            fund_nav_total = pickle.load(f)
        #导入股票收盘数据
        with open('./data/stock_daily_data.pickle','rb') as f:
            stock_daily_data = pickle.load(f)
        stock_industry = pd.read_excel('./data/stock_industry.xlsx')
        fund_scale = pd.read_excel('./data/fund_scale.xlsx')
        industry_index = pd.read_excel('./data/industry_index.xlsx', index_col = 0).pct_change(periods = 1)  
        fund_info = pd.read_excel('./data/fund_info.xlsx')
        self.fund_nav_total = fund_nav_total
        self.fund_position_total = fund_position_total
        self.stock_daily_data = stock_daily_data
        self.stock_industry = stock_industry
        self.fund_scale = fund_scale
        self.industry_index = industry_index  
        self.fund_info = fund_info

    #计算基金真实的持仓行业分布数据
    def fund_industry_weight(self, fund_code, date):
        fund_position = self.fund_position_total[fund_code]
        fund_position = fund_position[fund_position['end_date'] == date]
        scale = self.fund_scale[self.fund_scale['证券代码'] == fund_code][pd.to_datetime(date)].values[0]
        fund_position['stock_per_fund'] = fund_position['mkv'] / scale
        fund_position = fund_position.rename(columns = {'symbol':'证券代码'})
        fund_position_merge = self.stock_industry[['证券代码', pd.to_datetime(date)]].merge(fund_position[['证券代码', 'stock_per_fund']], on = '证券代码')
        fund_industry_weight = fund_position_merge.groupby(pd.to_datetime(date)).sum()
        fund_industry_weight.columns = [fund_code]
        fund_industry_weight = pd.concat([pd.DataFrame(self.industry_index.columns).set_index(0), fund_industry_weight], axis = 1)
        fund_industry_weight.loc['中债-国债总财富(总值)指数'] = 1 - fund_industry_weight.sum().values[0]
        return fund_industry_weight.fillna(0)
    
    #按照基金持仓股票编制行业指数
    def industry_makeup(self, fund_code, start_date, end_date):
        #获取持仓股票对应的行业
        fund_position = self.fund_position_total[fund_code]
        fund_position = fund_position[fund_position['end_date'] == start_date]
        scale = self.fund_scale[self.fund_scale['证券代码'] == fund_code][pd.to_datetime(start_date)].values[0]
        fund_position['stock_per_fund'] = fund_position['mkv'] / scale
        fund_position = fund_position.rename(columns = {'symbol':'证券代码'})
        fund_position_merge = self.stock_industry[['证券代码', pd.to_datetime(start_date)]].merge(fund_position[['证券代码', 'stock_per_fund']], on = '证券代码')
        
        #按照持仓股票制作行业指数
        industry_name = self.industry_index.columns.tolist()
        industry_index_makeup = pd.DataFrame()
        for i in range(len(industry_name)):
            
            code_weight = fund_position_merge[fund_position_merge[pd.to_datetime(start_date)] == industry_name[i]][['证券代码', 'stock_per_fund']]
            if code_weight.shape[0] != 0:
                code_weight['stock_per_fund'] = code_weight['stock_per_fund'] / code_weight['stock_per_fund'].sum()
                code_weight = code_weight.values
                industry_data = pd.DataFrame()
                for j in range(len(code_weight)):
                    stock_data = self.stock_daily_data[code_weight[j][0]]
                    stock_data = stock_data[(stock_data['trade_date'] >= start_date) & (stock_data['trade_date'] <= end_date)].sort_values(by = 'trade_date')[['trade_date', 'close']]
                    stock_data['close'] = (stock_data['close'].pct_change(periods = 1).fillna(0) + 1).cumprod() * code_weight[j][1]
                    industry_data = pd.concat([industry_data, stock_data.set_index('trade_date')], axis = 1).sum(axis = 1)
            elif code_weight.shape[0] == 0:
                industry_data = self.industry_index[(self.industry_index.index >= start_date) & (self.industry_index.index <= end_date)][[industry_name[i]]]
                industry_data.index = industry_data.index.strftime('%Y%m%d')
                industry_data.iloc[0] = 0
                industry_data = (industry_data + 1).cumprod()
            industry_index_makeup = pd.concat([industry_index_makeup, industry_data], axis = 1)
            
        industry_index_makeup.columns = industry_name
        industry_index_makeup.index = pd.to_datetime(industry_index_makeup.index)
        industry_index_makeup = industry_index_makeup.pct_change(periods = 1)  
        return industry_index_makeup
    
    #按照基金持仓股票编制行业指数并与中信一级指数加权
    def industry_makeup_mix(self, fund_code, start_date, end_date):
        industry_index_makeup = self.industry_makeup(fund_code, start_date, end_date)
        industry_index_extract = self.industry_index[(self.industry_index.index >= industry_index_makeup.index[0]) & (self.industry_index.index <= industry_index_makeup.index[-1])]
        return (industry_index_makeup + industry_index_extract) / 2
    
    #利用受约束的lasso回归进行基金仓位测算
    def lasso(self, fund_code, industry_index, start_date, end_date, lamb, stock_position_cons, time_weighted):
        '''数据准备'''        
        fund_index = self.fund_nav_total[fund_code][['nav_date', 'adj_nav']]
        fund_index['nav_date'] = fund_index['nav_date'].apply(lambda x: pd.to_datetime(x))
        fund_index = fund_index.drop_duplicates(subset = ['nav_date'], keep = 'first').set_index('nav_date').sort_index().pct_change(periods = 1)
        #数据筛选
        data = fund_index.join(industry_index).dropna().drop_duplicates()
        data_train = data[(data.index > pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))]
        Y = data_train['adj_nav'].values
        X = data_train.iloc[:, 1:].values   
        
        '''优化目标函数'''
        #目标函数定义 
        def target_equation(beta):
            if time_weighted == True:
                weight = np.array([np.sqrt(np.exp(i/(X.shape[0]))) for i in range(X.shape[0])])
                target_fun = np.multiply((Y - X @ beta), weight).T @ np.multiply((Y - X @ beta), weight) + lamb*sum(abs(beta))
            else:
                target_fun = (Y - X @ beta).T @ (Y - X @ beta) + lamb*sum(abs(beta))
            return target_fun
        
        #优化约束条件
        num = industry_index.shape[1]
        linear_constraint = LinearConstraint([list(np.ones(num)), list(np.concatenate((np.ones(num-1), np.array([0]))))], stock_position_cons[0], stock_position_cons[1])
        bounds = Bounds(list(np.zeros(num)), list(np.zeros(num)+1))
        beta0 = np.linalg.inv(X.T @ X) @ X.T @ Y
        #优化结果
        res = minimize(target_equation, beta0, method = 'SLSQP', constraints = [linear_constraint], bounds = bounds, options={'xtol': 1e-8, 'disp': True})
        fund_position_measure_res = pd.DataFrame(res.x, index = data_train.iloc[:, 1:].columns, columns = [fund_code]) / res.x.sum()
        return fund_position_measure_res
    
    #利用kalman滤波进行基金仓位测算
    def kalman(self, fund_code, industry_index, start_date, end_date, P0, Q, R): 
        '''数据准备''' 
        fund_index = self.fund_nav_total[fund_code][['nav_date', 'adj_nav']]
        fund_index['nav_date'] = fund_index['nav_date'].apply(lambda x: pd.to_datetime(x))
        fund_index = fund_index.drop_duplicates(subset = ['nav_date'], keep = 'first').set_index('nav_date').sort_index().pct_change(periods = 1)
        #数据筛选
        data = fund_index.join(industry_index).dropna().drop_duplicates()
        data_train = data[(data.index > pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))]           
        Y = data_train['adj_nav']
        X = data_train.iloc[:, 1:].values
        
        '''卡曼滤波'''
        #设置初始值
        res_true = self.fund_industry_weight(fund_code = fund_code, date = start_date)
        if res_true.shape[0] > 30:
            res_true = res_true.drop(['综合金融'])
        #进行卡曼滤波与平滑
        kalman = KalmanFilter()
        smooth = FixIntervalSmooth()
        #需要注意的是，这里的kalman fiter中的H是一个实时变化的矩阵；P0的选取对结果无影响；Q和R的选取取决于更信任观测方程还是预测方程
        kalman.KF(Z=Y, A=np.eye(X.shape[1]), H=X, state_names=data_train.columns[1:].tolist(), \
                  x0=res_true[fund_code].values, P0=P0, Q=Q, R=R)
        smooth.FIS(kalman)
        #参数归一化
        res_kalman = smooth.x_sm.iloc[-1:].T
        res_kalman[res_kalman < 0] = 0
        res_kalman = res_kalman / res_kalman.sum()        
        return res_kalman
    
    #对所有的偏股混合基金与普通股票型基金进行仓位测算
    def fund_industry_weight_measure(self, date, method):
        #筛选偏股混合基金与普通股票型基金
        fund_info = self.fund_info[(self.fund_info['基金类型'] == '普通股票型基金') | (self.fund_info['基金类型'] == '偏股混合型基金')]
        fund_info_extract = fund_info[fund_info['成立日期'] < pd.to_datetime(date) - pd.Timedelta(days=185)]
        fund_code_extract = fund_info_extract['证券代码'].tolist()
        #确定卡曼滤波的起始时间
        if int(date[4:6]) != 6 and int(date[4:6]) != 12:
            if 3 <= int(date[4:6]) < 8:
                index_begin_date = str(int(date[:4])-1) + '1231'
            elif (int(date[4:6]) >= 8):
                index_begin_date = str(int(date[:4])) + '0630'
            elif (int(date[4:6]) < 3):
                index_begin_date = str(int(date[:4])-1) + '0630'
        elif int(date[4:6]) == 6:
            index_begin_date = str(int(date[:4])-1) + '1231'
        elif int(date[4:6]) == 12:
            index_begin_date = str(int(date[:4])) + '0630'
            
        #进行筛选出的基金仓位测算
        measure_res_all = pd.DataFrame()
        for i in trange(len(fund_code_extract)):
            if fund_info.iloc[10]['基金类型'] == '偏股混合型基金':
                stock_weight_low = 0.6
            elif fund_info.iloc[10]['基金类型'] == '普通股票型基金':
                stock_weight_low = 0.8
            try:
                if method == 'kalman_makeup':
                    measure_res = self.kalman(fund_code = fund_code_extract[i], industry_index = self.industry_makeup(fund_code_extract[i], index_begin_date, date), start_date = index_begin_date, end_date = date, P0=np.ones([30,30])*1e-10, Q=np.eye(30)*1e-5, R=1e-6)
                elif method == 'kalman_mix':
                    measure_res = self.kalman(fund_code = fund_code_extract[i], industry_index = self.industry_makeup_mix(fund_code_extract[i], index_begin_date, date), start_date = index_begin_date, end_date = date, P0=np.ones([30,30])*1e-10, Q=np.eye(30)*1e-5, R=1e-6)
                elif method == 'kalman_origin':
                    measure_res = self.kalman(fund_code = fund_code_extract[i], industry_index = self.industry_index, start_date = index_begin_date, end_date = date, P0=np.ones([30,30])*1e-10, Q=np.eye(30)*1e-5, R=1e-6)
                    
                elif method == 'lasso_mix':
                    measure_res = self.lasso(fund_code = fund_code_extract[i], industry_index = self.industry_makeup_mix(fund_code_extract[i], index_begin_date, date), start_date = (pd.to_datetime(date) - pd.Timedelta(days=40)).strftime('%Y%m%d'), end_date = date, lamb = 3e-6, stock_position_cons = [[1, stock_weight_low], [1, 0.95]], time_weighted = True)
                elif method == 'lasso_makeup':
                    measure_res = self.lasso(fund_code = fund_code_extract[i], industry_index = self.industry_makeup(fund_code_extract[i], index_begin_date, date), start_date = (pd.to_datetime(date) - pd.Timedelta(days=40)).strftime('%Y%m%d'), end_date = date, lamb = 3e-6, stock_position_cons = [[1, stock_weight_low], [1, 0.95]], time_weighted = True)
                elif method == 'lasso_origin':
                    measure_res = self.lasso(fund_code = fund_code_extract[i], industry_index = self.industry_index, start_date = (pd.to_datetime(date) - pd.Timedelta(days=40)).strftime('%Y%m%d'), end_date = date, lamb = 3e-6, stock_position_cons = [[1, stock_weight_low], [1, 0.95]], time_weighted = True)
                measure_res.columns = [fund_code_extract[i]]
                measure_res_all = pd.concat([measure_res_all, measure_res], axis = 1)
            except:
                pass
        self.measure_res_all = measure_res_all
        #对持仓进行规模加权平均
        fund_scale_extract = self.fund_scale[['证券代码', pd.to_datetime(index_begin_date)]]
        fund_scale_extract = fund_scale_extract[fund_scale_extract['证券代码'].isin(measure_res_all.columns.tolist())]
        fund_scale_extract[pd.to_datetime(index_begin_date)] = fund_scale_extract[pd.to_datetime(index_begin_date)] / fund_scale_extract.sum().values[1]
        res = pd.DataFrame([fund_scale_extract.iloc[:,1].tolist() for i in range(30)], columns = fund_scale_extract['证券代码'].tolist(), index = measure_res_all.index).T.multiply(measure_res_all.T)
        res = res.sum().sort_values(0, ascending = False)
        res.columns = [pd.to_datetime(date)]
        return res  
    
    #统计全市场偏股混合基金与普通股票基金的半年报年报持仓情况
    def fund_industry_weight_true(self, date):
        fund_info = self.fund_info[(self.fund_info['基金类型'] == '普通股票型基金') | (self.fund_info['基金类型'] == '偏股混合型基金')]
        fund_info_extract = fund_info[fund_info['成立日期'] < pd.to_datetime(date) - pd.Timedelta(days=185)]
        fund_code_extract = fund_info_extract['证券代码'].tolist()
        true_res_all = pd.DataFrame()
        for i in trange(len(fund_code_extract)):
            try:
                true_res = self.fund_industry_weight(fund_code_extract[i], date)
                true_res.columns = [fund_code_extract[i]]
                true_res_all = pd.concat([true_res_all, true_res], axis = 1).dropna()
            except:
                pass
        self.true_res_all = true_res_all
        #对持仓进行规模加权平均
        fund_scale_extract = self.fund_scale[['证券代码', pd.to_datetime(date)]]
        fund_scale_extract = fund_scale_extract[fund_scale_extract['证券代码'].isin(true_res_all.columns.tolist())]
        fund_scale_extract[pd.to_datetime(date)] = fund_scale_extract[pd.to_datetime(date)] / fund_scale_extract.sum().values[1]
        res = pd.DataFrame([fund_scale_extract.iloc[:,1].tolist() for i in range(30)], columns = fund_scale_extract['证券代码'].tolist(), index = true_res_all.index).T.multiply(true_res_all.T)
        res = res.sum().sort_values(0, ascending = False)
        res.columns = [pd.to_datetime(date)]
        return res