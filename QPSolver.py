#!/usr/bin/env python
# -- coding:utf-8 --

from support import * 
# import FeatureVector

class QPSolver(object):
    """
    solve quadratic programming problem with hildreth algorithm
    Borrowed from <a  href="http://www.seas.upenn.edu/~strctlrn/StructLearn/StructLearn.html">
    Penn StructLearn </a>
    """
    MAX_ITER = 10000
    EPS = 1e-8
    ZERO = 1e-16
    def hildreth(self, fv, loss):
        '''迭代求解alpha参数数组
        Args:
            fv:特征向量数组，是两个向量之间差分的结果
            loss:损失分数数组，是一个数据集每个特征向量和最佳特征向量的评分损失值
        Return:
            alpha：一组参数
        Raise:
            None
        '''
        LengthOfb = loss.__len__()
        alpha = [0.0] * LengthOfb
        F = [0.0] * LengthOfb
        kkt = [0.0] * LengthOfb
        # GramMatrix用于缓存向量数组的內积，用于降低时间复杂度
        K = fv.__len__()
        GramMatrix = [[0] * K for i in range(K)]
        is_computed = [False] * K
        for i in range(K):
            GramMatrix[i][i] = FeatureVector().dotProduct(fv[i],fv[i])
        # 寻找loss数组中最大数组项及其索引
        max_kkt = float("-inf")
        max_kkt_i = -1
        for i in range(LengthOfb):
            F[i] = loss[i]
            kkt[i] = F[i]
            if kkt[i] > max_kkt:
                max_kkt = kkt[i]
                max_kkt_i = i
        circle = 0
        diff_alpha = 0.0
        try_alpha = 0.0
        add_alpha = 0.0
        while max_kkt >= self.EPS and circle < self.MAX_ITER:
            # 更新loss最大项的alpha值
            diff_alpha = F[max_kkt_i] / GramMatrix[max_kkt_i][max_kkt_i]
            if GramMatrix[max_kkt_i][max_kkt_i] <= self.ZERO:
                diff_alpha = 0.0
            try_alpha = alpha[max_kkt_i] + diff_alpha
            if try_alpha < 0.0:
                add_alpha = -1.0 * alpha[max_kkt_i]
            else:
                add_alpha = diff_alpha
            alpha[max_kkt_i] += add_alpha
            # 提前计算好所用的向量內积
            if not is_computed[max_kkt_i]:
                for i in range(K):
                    GramMatrix[i][max_kkt_i] = FeatureVector().dotProduct(fv[i],fv[max_kkt_i])
                    is_computed[max_kkt_i] = True
            for i in range(LengthOfb):
                F[i] -= add_alpha * GramMatrix[i][max_kkt_i]
                kkt[i] = F[i]
                if alpha[i] > self.ZERO:
                    kkt[i] = abs(F[i])
            # 每次迭代都处理loss最大项
            max_kkt = float("-inf")
            max_kkt_i = -1
            for i in range(LengthOfb):
                if kkt[i] > max_kkt:
                    max_kkt = kkt[i]
                    max_kkt_i = i
            circle += 1
        return alpha

def main():
    qp = QPSolver()
    print qp.hildreth([{1:1.0,2:1.0,3:1.0},{2:1.0,3:1.0,6:1.0}],[10,11])
if __name__ == '__main__':
    main()