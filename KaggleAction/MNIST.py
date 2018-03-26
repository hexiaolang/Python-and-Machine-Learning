# -*- coding:utf-8 -*-
import pandas as pd
import skflow

train = pd.read_csv('../Datasets/MNIST/train.csv')  # 42000*785
test = pd.read_csv('../Datasets/MNIST/test.csv')  # 28000*784

y_train = train['label']
X_train = train.drop('label', 1)
X_test = test

# 使用skflow中已经封装好的基于TensorFlow搭建的线性分类器
# TensorFlowLinearClassifier进行学习预测
classifier = skflow.TensorFlowLinearClassifier(n_classes=10,
                                               batch_size=100,
                                               steps=1000,
                                               learning_rate=0.01)
classifier.fit(X_train, y_train)
linear_y_pred = classifier.predict(X_test)
linear_submission = pd.DataFrame({'ImageId': range(1, 28001),
                                  'Label':linear_y_pred})
linear_submission.to_csv('../Datasets/MNIST/linear_submission.csv',
                         index=False)

# 基于TensorFlow搭建的全连接深度神经网络TensorFlowDNNClassifier进行
# 学习预测
classifier = skflow.TensorFlowDNNClassifier(
    hidden_units=[200, 50, 10], n_classes=10, steps=5000,
    learning_rate=0.01, batch_size=50)
classifier.fit(X_train, y_train)
dnn_y_pred = classifier.predict(X_test)
dnn_submission = pd.DataFrame({'ImageId': range(1, 28001),
                               'Label': dnn_y_pred})
dnn_submission.to_csv('../Datasets/MNIST/dnn_submission.csv',
                      index=False)

# 使用TensorFlow中的算子自行搭建更为复杂的卷积神经网络，并使用skflow的
# 程序接口从事MNIST数据的学习与预测
# def max_pool_2x2(tensor_in):
#     return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1],
#                           strides=[1, 2, 2, 1],
#                           padding='SAME')
# def conv_model(X, y):
#     X = tf.reshap(X, [-1, 28, 28, 1])