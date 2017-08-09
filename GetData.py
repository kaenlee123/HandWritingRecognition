# encoding: utf-8

"""
@version: python3.5.2
@author: kaenlee  @contact: lichaolfm@163.com
@software: PyCharm Community Edition
@time: 2017/8/7 23:38
purpose:
"""

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(r'data', one_hot=True)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
print(mnist.validation.images.shape)
print(mnist.validation.labels.shape)
print(mnist.test.images.shape)
print(mnist.test.labels.shape)