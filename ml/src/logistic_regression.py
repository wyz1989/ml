#!/usr/bin/python
#encoding:utf8
"""
@author: yizhong.wyz
@date: 2018-05-03
@desc: 逻辑回归
"""
import re
import json
from numpy import *
from random import shuffle

import tlib.log as log



def sigmoid(inx):
    return 1.0 / (1 + exp(-inx))


def batchMatrix(dataMat, labelMat, batch_size):
    """
    获取batch 矩阵
    """
    batchMat = []
    batchLabelMat = []
    index = range(len(dataMat))
    shuffle(index)
    for i in range(batch_size):
        batchMat.append(dataMat[index[i]])
        batchLabelMat.append(labelMat[index[i]])
    return mat(batchMat), mat(batchLabelMat).transpose()
   

class LogicRegression(object):
    def __init__(self):
        self.dataMat = []
        self.labelMat = []
    
    def loadDataSet(self, filename='../data/testSet.txt'):
        """
        加载训练数据
        """
        f = open(filename, "r")
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            tmp_list = re.split(r"\s+", line)
            if len(tmp_list) != 3:
                log.error("train data line is wrong")
                continue
            self.dataMat.append([1.0, float(tmp_list[0]), float(tmp_list[1])])
            self.labelMat.append(int(tmp_list[2]))

    def gradAscent(self, alpha=0.001, iter_times=100):
        """
        梯度下降法求权重
        损失函数取平方损失函数 sum((h(x) - y)**2) / 2
        @alpha 训练速率
        @iter_times 迭代次数
        """
        dataMatrix = mat(self.dataMat)
        labelMat = mat(self.labelMat).transpose()
        m, n = shape(dataMatrix)
        weights = ones((n, 1))
        for k in range(iter_times):
            h = sigmoid(dataMatrix*weights)
            error = (h - labelMat)
            weights -= alpha*dataMatrix.transpose() * error
        return weights

    def gradAscentBatch(self, alpha=0.001, iter_times=100, batch_size=5):
        """
        随机梯度下降算法
        动态更新学习速率
        @batch_size  随机抽batch_size个样本进行权重更新
        """
        dataMatrix = mat(self.dataMat)
        labelMat = mat(self.labelMat).transpose()
        m, n = shape(dataMatrix)
        weights = ones((n, 1))
        for k in range(iter_times):
            alpha = 4 / (k + 5) + 0.001
            tmp_data_mat, tmp_label_mat = batchMatrix(self.dataMat, self.labelMat, batch_size)
            h = sigmoid(tmp_data_mat * weights)
            error = h - tmp_label_mat
            weights -= alpha*tmp_data_mat.transpose()*error
        return weights
            

if __name__ == "__main__":
    lr = LogicRegression()
    lr.loadDataSet()
    weights = lr.gradAscentBatch()
    log.info("f(x)=%s+%sx1+%sx2" % (weights[0][0], weights[1][0], weights[2][0]))
