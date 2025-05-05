"""
Interpolation of image data using multiple site data

@author: Ty
"""
# coding:utf-8

import os
import sys
import time

import numpy as np
import pandas as pd
import PySimpleGUI as sg


def length_2_point(x11, y1, x22, y2):
    return ((x11 - x22) ** 2 + (y1 - y2) ** 2) ** 0.5


def get_Several_MinMax_Array(np_arr, several):
    """
    获取numpy数值中最大或最小的几个数
    :param np_arr:  numpy数组
    :param several: 最大或最小的个数（负数代表求最大，正数代表求最小）
    :return:
        several_min_or_max: 结果数组
    """
    if several > 0:
        several_min_or_max = np_arr[np.argpartition(np_arr, several)[:several]]
    else:
        several_min_or_max = np_arr[np.argpartition(np_arr, several)[several:]]
    return several_min_or_max


def Idw(data, loc_, size=6, name='处理中'):
    # 计算权重矩阵
    all_w = [[], [], []]  # 权重顺序：郴州，连州，韶关
    for i_ in range(1, size + 1):
        for j in range(1, size + 1):
            l_f = []
            for k in loc_:
                l_f.append(length_2_point(i_, j, k[0], k[1]))
            l_f = np.asarray(l_f)
            st = np.argsort(l_f)
            if l_f[st[0]] == 0:
                all_w[st[0]].append(1)
                all_w[st[1]].append(0)
                all_w[st[2]].append(0)
            else:
                total_length = 1 / l_f[st[0]] + 1 / l_f[st[1]] + 1 / l_f[st[2]]
                all_w[st[0]].append(1 / l_f[st[0]] / total_length)
                all_w[st[1]].append(1 / l_f[st[1]] / total_length)
                all_w[st[2]].append(1 / l_f[st[2]] / total_length)
    all_w = np.asarray(all_w)
    all_w = np.reshape(all_w, newshape=[-1, size, size])
    pic = []
    for n, one in enumerate(data):
        sg.one_line_progress_meter(name + '插值进度条', n + 1, len(data), '-key-')
        chen_zhou = np.ones((size, size)) * one[0]
        lian_zhou = np.ones((size, size)) * one[1]
        shao_guan = np.ones((size, size)) * one[2]
        pic.append(chen_zhou * all_w[0] + lian_zhou * all_w[1] + shao_guan * all_w[2])

    np.save(r"data/" + name + str(size) + "×" + str(size) + ".npy", np.asarray(pic))
    return np.asarray(pic)


if __name__ == '__main__':
    data_i = pd.read_csv(r"data/2012.7_2015.4_6h.csv")
    factors = ["prec", "tem"]
    ch_fac = ["降雨", "气温"]
    site = ["cz_", "lz_", "sg_"]

    # loc = np.array([[1, 3], [5, 1], [5, 5]])
    # loc = np.array([[2, 5], [8, 1], [9, 9]])
    loc = np.array([[1, 4], [6, 1], [6, 6]])
    for i, fac in enumerate(factors):
        print("共插值{}个因子，现在是第{}个：{}".format(len(factors), i + 1, ch_fac[i]))
        data_ = np.concatenate([np.expand_dims(data_i[site[0] + fac].values, axis=1),
                                np.expand_dims(data_i[site[1] + fac].values, axis=1),
                                np.expand_dims(data_i[site[2] + fac].values, axis=1),
                                ], axis=1)
        Idw(data_, loc, name=fac)
    print("\nAll completed.")
