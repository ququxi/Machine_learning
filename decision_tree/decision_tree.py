import pandas as pd
import numpy as np
from collections import Counter

## 信息熵
def entropy(Xv):
    cnt = pd.Series(Counter(Xv))
    return -np.sum((cnt/cnt.sum()) * np.log2(cnt/cnt.sum()))

## 信息增益(属性Xvar)
def Info_gain(xVar,data,target = 'isgood'):
    init_val = 0
    for item in data[xVar].unique():
        init_val += entropy(data[data[xVar] == item][target])*(data[data[xVar] == item].shape[0]/data.shape[0])
    return entropy(data[target]) - init_val

## 信息增益率
def Gain_ratio(xVar,data,target = 'isgood'):
    return Info_gain(xVar,data = data,target = target)/entropy(data[xVar])

def gini(Xv):
    cnt = pd.Series(Counter(Xv))
    return 1 - np.sum((cnt/cnt.sum())*(cnt/cnt.sum()))

## 基尼系数
def gini_index(xVar,data,target = 'isgood'):
    init_val = 0
    for item in data[xVar].unique():
        init_val += gini(data[data[xVar]==item][target])*(data[data[xVar] == item].shape[0]/data.shape[0])
    return init_val

def continuous_gain(xVar,data,target = 'isgood'):
    '''
    data: source数据源
    target:定义目标变量
    xVar:变量名，计算相关变量的信息增益值，其类型为连续性数值变量
    其背后逻辑详见：周志华《机器学习》P84
    '''
    val = pd.Series(data[xVar].unique())
    sorta = val.sort_values()
    Ta = (sorta.shift(-1)+sorta)/2
    res = 0
    resA = None
    for t in Ta:
        data['nw'] = '大于临界值'
        data.loc[data[xVar] < t,'nw'] = '小于临界值'
        gain = Info_gain(xVar = 'nw',data = data,target = target)
        if res < gain:
            res = gain
            resA = t
    return res,resA