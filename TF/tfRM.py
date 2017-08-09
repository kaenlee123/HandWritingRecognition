# encoding: utf-8

"""
@version: python3.5.2 
@author: kaenlee  @contact: lichaolfm@163.com
@software: PyCharm Community Edition
@time: 2017/8/7 23:38
purpose:http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)

# 占位符
x = tf.placeholder('float', [None, 784])

# 张量
W = tf.Variable(tf.zeros(shape=[784, 10], dtype=tf.float32, name='W'))  # 274维度输入， 10个输出label
b = tf.Variable(tf.zeros(shape=[10], dtype=tf.float32, name='b'))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder('float', [None, 10])

# 计算交叉熵:y y预测值， y_真实值
crossEntropy = -tf.reduce_sum(y_ * tf.log(y))

# 反向传播计算最佳w b
trainStep = tf.train.GradientDescentOptimizer(0.01).minimize(crossEntropy)

# 运行之前创建初始变量
init = tf.initialize_all_variables()

# 在Session启动模型
sess = tf.Session()
sess.run(init)

# 训练模型
for i in range(1000):
    # mnist 随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(trainStep, feed_dict={x: batch_xs, y_: batch_ys})
print(sess.run(W), sess.run(b))


#下面对模型进行评价
# 判断预测值得和真实值1所在的位置是否一致:boolean
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# 转换成float然后取平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))  # 用测试数据进行检验