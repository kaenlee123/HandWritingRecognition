# encoding: utf-8

"""
@version: python3.5.2 
@author: kaenlee  @contact: lichaolfm@163.com
@software: PyCharm Community Edition
@time: 2017/8/9 13:07
purpose:
"""
import numpy as np

class NetWorks:
    # 定义一个神经网络，也就是定义每一层的权重以及偏置
    def __init__(self, size):
        """
        给出每层的节点数量，包含输出输出层
        :param size: list
        """
        self.size = size
        self.Layers = len(size)
        # 以正太分布形式随机赋予初始值
        # 每一层的偏置以行保存在list
        self.bias = [np.mat(np.random.randn(num)).T for num in size[1:]]  # 输入层没有bias
        # 每层的权重取决于row取决于该层的节点数量，从来取决于前面一层的输出即节点数
        self.weight = [np.mat(np.random.randn(row, col)) for row, col in zip(size[1:], size[:-1])]

    def Sigmod(self, z):
        return 1 / np.exp(-z)

    def feedward(self, a):
        """
        d对网络给定输入，输出对应的输出
        :param a:@iterable给定的输入向量
        :return:
        """
        # 上层m节点， 当前n个
        # 每一层的输出: W * X.T + b
        # W[n, m]
        # X: 为上一层(节点数m)
        # 的输出[1, m]
        # b: [n, 1]
        a = np.mat(a).T
        for b, w in zip(self.bias, self.weight):
            linear = w * a + b  # 返回一个nX1的矩阵
            a = self.Sigmod(linear)
        return a.T



if __name__ == '__main__':
    net = NetWorks([3, 3, 2])
    print(net.feedward([1, 1, 1]))

