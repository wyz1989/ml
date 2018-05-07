#!/usr/bin/python
#encoding:utf8
"""
@author: yizhong.wyz
@date: 2018-05-03
@desc: 最大熵模型
"""

import re
import math
from collections import defaultdict

import tlib.log as log

class MaxEntropy(object):
    def __init__(self):
        self.sample = []    #样本
        self.N = 0          #样本个数
        self.numXY = defaultdict(int)   #(x,y) 频次统计
        self.n = 0          #特征(xi, yi)统计
        self.Y = set([])    #类别set
        self.W = []         #权重
        self.last_W = []    #记录前一次权重
        self.C = 0          #样本特征最大个数
        self.ep = []        #模型分布的特征期望值
        self.ep_ = []       #样本分布的特征期望值
        self.xyID = defaultdict(int)
        self.EPS = 0.01     #判断权重收敛
    
    def load_data(self, filename="../data/data.txt"):
        """
        加载训练数据,样本格式: label feature1 featrue2 feature3 ....
        """
        lines = open(filename, "r").readlines()
        for line in lines:
            line = line.strip("\n")
            sample = re.split(r"\s+", line)
            if len(sample) < 2:
                continue
            y = sample[0]
            X = sample[1:]
            self.sample.append(sample)
            self.Y.add(y)
            for x in set(X):
                self.numXY[(x, y)] += 1
   
    def _init_params(self):
        """
        参数初始化
        """
        self.N = len(self.sample)
        self.n = len(self.numXY)
        self.C = max([len(X) -1 for X in self.sample])
        self.W = [0.0] * self.n
        self.last_W = [0.0] * self.n
        self._example_ep()

    def _example_ep(self):
        """
        计算经验概率分布对特征函数f(x,y)的期望，公式如下:
        E~p(f) = sum_xy(_p(x,y)*f(x,y))
        """
        self.ep_ = [0.0] * self.n
        for i, xy in enumerate(self.numXY):
            self.ep_[i] = self.numXY[xy] * 1.0 / self.N
            self.xyID[xy] = i
    
    def _zx(self, X):
        """
        z(x) = sum_y(exp(sum_n(wi*fi(x,y))))
        """
        ZX = 0.0
        for y in self.Y:
            sum = 0.0
            for x in X:
                if (x, y) in self.numXY:
                    sum += self.W[self.xyID[(x, y)]]
            ZX += math.exp(sum)
        return ZX

    def _pyx(self, X):
        """
        根据最大熵模型求条件概率分布
        p(y|x) = exp(sum_n(wi*fi(x,y))) / Z(x)
        """
        ZX = self._zx(X)
        results = []
        for y in self.Y:
            sum = 0.0 
            for x in X:
                if (x, y) in self.numXY:
                    sum += self.W[self.xyID[(x, y)]]
            pyx = 1.0 / ZX * math.exp(sum)
            results.append((y, pyx))
        return results

    def _model_ep(self):
        """
        条件概率p(y|x) 经验分布_p(x)对特征函数f(x,y)的期望
        Ep(f) = sum_xy(_p(x)*p(y|x)*f(x,y))
        """
        self.ep = [0.0] * self.n
        for sample in self.sample:
            X = sample[1:]
            pyx = self._pyx(X)
            for y, p in pyx:
                for x in X:
                    if (x, y) in self.numXY:
                        self.ep[self.xyID[(x, y)]] += p*1.0 / self.N

    def _check_convergence(self):
        """
        检查算法收敛
        """
        for w, lw in zip(self.W, self.last_W):
            if math.fabs(w - lw) >= self.EPS:
                return False
        return True


    def _gis(self, maxiter):
        """
        迭代尺度算法
        """
        self._init_params()
        for i in range(maxiter):
            log.info("Iter:%s............." % i)
            self.last_W = self.W[:]
            self._model_ep()
            ###权重更新
            for j, w in enumerate(self.W):
                self.W[j] += 1.0 / self.C * math.log(self.ep_[j] / self.ep[j])
            log.info("W:%s" % self.W)
            if self._check_convergence():
                break

    def train(self, maxiter=1000, alg_type="gis"):
        """
        @maxiter 最大迭代次数
        @alg_type 算法选择 [gis|] gis迭代尺度算法
        """
        if alg_type == "gis":
            self._gis(maxiter)

    def predict(self, sample):
        X = re.split(r"\s+", sample)
        p = self._pyx(X)
        return p


if __name__ == "__main__":
    maxent = MaxEntropy()
    maxent.load_data()
    maxent.train(maxiter=1000)
    p = maxent.predict("sunny hot high FALSE")
    log.info("predict:%s" % p)



