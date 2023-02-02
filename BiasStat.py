# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 01:57:51 2023

@author: Hangyu
"""

from PositionMeasure import *

def bias_stat(date_all, method):
    with open('./res/' + method + '.pickle','rb') as f:
        method_res = pickle.load(f)
    with open('./res/' + method + '_detail' + '.pickle','rb') as f:
        method_detail = pickle.load(f)
    with open('./res/res_true_all.pickle','rb') as f:
        res_true_all = pickle.load(f)        
    with open('./res/res_true_detail.pickle','rb') as f:
        res_true_detail = pickle.load(f)  
        
    measure_res = pd.DataFrame()
    true_res = pd.DataFrame()
    single_fund_bias_total = []
    for i in range(len(date_all) - 2):
        date = date_all[i+1].strftime('%Y%m%d')
        df_measure = pd.DataFrame(method_res[date])
        df_measure.columns = [date]
        df_true = pd.DataFrame(res_true_all[pd.to_datetime(date)])
        df_true.columns = [date]
        measure_res = pd.concat([measure_res, df_measure], axis = 1)
        true_res = pd.concat([true_res, df_true], axis = 1)
    
        single_fund_bias_total.append([date, abs(method_detail[date] - res_true_detail[pd.to_datetime(date)][method_detail[date].columns]).sum().sum() / 29 / method_detail[date].shape[1]])
    
    bias_stat_res = pd.DataFrame(abs(true_res - measure_res).sum() / 29).join(pd.DataFrame(single_fund_bias_total).set_index(0))
    bias_stat_res.columns = ['all_fund_bias', 'sigle_fund_bias']
    return bias_stat_res

if __name__ == '__main__':
    measure = fund_position_measure()
    date_all = pd.date_range('20180630', '20221231', freq = '2Q')
    
    #获取各年度的年报中报估计结果
    '''
        for i in range(len(date_all)-1):
            print([date_all[i], date_all[i+1]])
        
        kalman_origin_detail = {}
        kalman_origin = {}
        kalman_makeup_detail = {}
        kalman_makeup = {}
        kalman_mix_detail = {}
        kalman_mix = {}
        
        lasso_origin_detail = {}
        lasso_origin = {}
        lasso_mix_detail = {}
        lasso_mix = {}
        lasso_makeup_detail = {}
        lasso_makeup = {}
        
        res_true_all = {}
        res_true_detail = {}
        for i in trange(len(date_all)-1):
        
            res_kalman_mix = measure.fund_industry_weight_measure(date_all[i].strftime('%Y%m%d'), date_all[i+1].strftime('%Y%m%d'), 'kalman_mix')
            kalman_mix[date_all[i+1].strftime('%Y%m%d')] = res_kalman_mix
            kalman_mix_detail[date_all[i+1].strftime('%Y%m%d')] = measure.measure_res_all
            
            res_kalman_makeup = measure.fund_industry_weight_measure(date_all[i].strftime('%Y%m%d'), date_all[i+1].strftime('%Y%m%d'), 'kalman_makeup')
            kalman_makeup[date_all[i+1].strftime('%Y%m%d')] = res_kalman_makeup
            kalman_makeup_detail[date_all[i+1].strftime('%Y%m%d')] = measure.measure_res_all
        
            res_kalman = measure.fund_industry_weight_measure(date_all[i].strftime('%Y%m%d'), date_all[i+1].strftime('%Y%m%d'), 'kalman_origin')
            kalman_origin[date_all[i+1].strftime('%Y%m%d')] = res_kalman
            kalman_origin_detail[date_all[i+1].strftime('%Y%m%d')] = measure.measure_res_all
        
            res_lasso_mix = measure.fund_industry_weight_measure(date_all[i].strftime('%Y%m%d'), date_all[i+1].strftime('%Y%m%d'), 'lasso_mix')
            lasso_mix[date_all[i+1].strftime('%Y%m%d')] = res_lasso_mix
            lasso_mix_detail[date_all[i+1].strftime('%Y%m%d')] = measure.measure_res_all
        
            res_lasso_makeup = measure.fund_industry_weight_measure(date_all[i].strftime('%Y%m%d'), date_all[i+1].strftime('%Y%m%d'), 'lasso_makeup')
            lasso_makeup[date_all[i+1].strftime('%Y%m%d')] = res_lasso_makeup
            lasso_makeup_detail[date_all[i+1].strftime('%Y%m%d')] = measure.measure_res_all
        
            res_lasso = measure.fund_industry_weight_measure(date_all[i].strftime('%Y%m%d'), date_all[i+1].strftime('%Y%m%d'), 'lasso_origin')
            lasso_origin[date_all[i+1].strftime('%Y%m%d')] = res_lasso
            lasso_origin_detail[date_all[i+1].strftime('%Y%m%d')] = measure.measure_res_all
        
            res_true = measure.fund_industry_weight_true(date_all[i+1].strftime('%Y%m%d'))
            res_true_all[date_all[i+1].strftime('%Y%m%d')] = res_true
            res_true_detail[date_all[i+1].strftime('%Y%m%d')] = measure.true_res_all
        
        with open('./res/kalman_mix.pickle','wb') as f:
            pickle.dump(kalman_mix, f)
            f.close()
        
        with open('./res/kalman_mix_detail.pickle','wb') as f:
            pickle.dump(kalman_mix_detail, f)
            f.close()
        
        with open('./res/kalman_makeup.pickle','wb') as f:
            pickle.dump(kalman_makeup, f)
            f.close()
        
        with open('./res/kalman_makeup_detail.pickle','wb') as f:
            pickle.dump(kalman_makeup_detail, f)
            f.close()
        
        with open('./res/kalman_origin.pickle','wb') as f:
            pickle.dump(kalman_origin, f)
            f.close()
        
        with open('./res/kalman_origin_detail.pickle','wb') as f:
            pickle.dump(kalman_origin_detail, f)
            f.close()
            
            
        with open('./res/lasso_origin.pickle','wb') as f:
            pickle.dump(lasso_origin, f)
            f.close()
        
        with open('./res/lasso_origin_detail.pickle','wb') as f:
            pickle.dump(lasso_origin_detail, f)
            f.close()
        
        with open('./res/lasso_mix_detail.pickle','wb') as f:
            pickle.dump(lasso_mix_detail, f)
            f.close()
        
        with open('./res/lasso_mix.pickle','wb') as f:
            pickle.dump(lasso_mix, f)
            f.close()
        
        with open('./res/lasso_makeup_detail.pickle','wb') as f:
            pickle.dump(lasso_makeup_detail, f)
            f.close()
        
        with open('./res/lasso_makeup.pickle','wb') as f:
            pickle.dump(lasso_makeup, f)
            f.close()
        
        with open('./res/res_true_all.pickle','wb') as f:
            pickle.dump(res_true_all, f)
            f.close()
            
        with open('./res/res_true_detail.pickle','wb') as f:
            pickle.dump(res_true_detail, f)
            f.close()
    '''
    #统计单个基金的行业偏差绝对值的均值与总体市场的行业偏差绝对值的均值
    kalman_mix_bias = bias_stat(date_all, 'kalman_mix')
    kalman_makeup_bias = bias_stat(date_all, 'kalman_makeup')
    kalman_origin_bias = bias_stat(date_all, 'kalman_origin')
    
    lasso_mix_bias = bias_stat(date_all, 'lasso_mix')
    lasso_makeup_bias = bias_stat(date_all, 'lasso_makeup')
    lasso_origin_bias = bias_stat(date_all, 'lasso_origin')
    
    #将统计结果进行汇总，all_fund_bias代表全市场的统计结果，single_fund_bias代表单个基金平均偏差的平均值，单位为%
    bias_stat_res = pd.concat([pd.DataFrame(lasso_origin_bias.mean(), columns = ['lasso_origin_bias']), pd.DataFrame(lasso_makeup_bias.mean(), columns = ['lasso_makeup_bias']),
                    pd.DataFrame(lasso_mix_bias.mean(), columns = ['lasso_mix_bias']), pd.DataFrame(kalman_origin_bias.mean(), columns = ['kalman_origin_bias']),
                    pd.DataFrame(kalman_makeup_bias.mean(), columns = ['kalman_makeup_bias']), pd.DataFrame(kalman_mix_bias.mean(), columns = ['kalman_mix_bias'])], axis = 1) * 100
    '''
    with open('./res/kalman_mix.pickle','rb') as f:
        kalman_mix = pickle.load(f)

    with open('./res/kalman_mix_detail.pickle','rb') as f:
        kalman_mix_detail = pickle.load(f)
        
    with open('./res/res_true_all.pickle','rb') as f:
        res_true_all = pickle.load(f)
        
    date = '20200630'
    kalman_mix_res = kalman_mix[date]
    true_res = measure.fund_industry_weight_true(date)
    kalman_mix_res = kalman_mix_res[kalman_mix_res.index != '中债-国债总财富(总值)指数']
    true_res = true_res[true_res.index != '中债-国债总财富(总值)指数']
    res_merge = pd.concat([true_res, kalman_mix_res], axis = 1)
    res_merge.columns = ['真实数据', '测算数据']
    
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.figure(figsize=(8,4),dpi = 750)
    plt.plot(res_merge.iloc[:,0], label = res_merge.columns[0])
    plt.plot(res_merge.iloc[:,1], label = res_merge.columns[1])
    plt.legend()
    plt.xticks(rotation=90) 
    plt.title(date)
    
    date_all = pd.date_range('20151231', '20220630', freq = '2Q')
    res_true_detail_all = pd.DataFrame()
    kalman_mix_detail_all = pd.DataFrame()
    for i in range(len(date_all)):
        res_true_detail_all = pd.concat([res_true_detail_all, pd.DataFrame(res_true_all[date_all[i]], columns = [date_all[i].strftime('%Y%m%d')])], axis = 1)
        kalman_mix_detail_all = pd.concat([kalman_mix_detail_all, pd.DataFrame(kalman_mix[date_all[i].strftime('%Y%m%d')], columns = [date_all[i].strftime('%Y%m%d')])], axis = 1)
    
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif']=['SimHei']  
    plt.figure(figsize=(8,4),dpi = 750)
    plt.plot(res_true_detail_all.loc['煤炭'].plot(), label = '真实数据')
    plt.plot(kalman_mix_detail_all['煤炭'].plot(), label = '测算数据')
    plt.legend()
    '''