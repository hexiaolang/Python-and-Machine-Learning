# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

# 创建特征列表
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
               'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
               'Normal Nucleoli', 'Mitoses', 'cLass']

# 使用pandas.read_csv从互联网读取指定数据
data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
                   names=column_names)
# 将?替换为标准缺失值表示
data = data.replace(to_replace='?', value=np.nan)
# 丢弃带有缺失值的数据（只要有一个维度有缺失）
data = data.dropna(how='any')
# 使用sklearn.cross_validation里的train_test_split模块用于分割数据
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]],
                                                    data[column_names[10]],
                                                    test_size=0.25,
                                                    random_state=33)
# 标准化数据，保证每个维度的特征数据方差为1，均值为0.使得预测结果不会被某些维度过大的特征值而主导
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
# 初始化LogisticRegression与SGDClassifier
lr = LogisticRegression()
sgdc = SGDClassifier()
# 调用LogisticRegression中的fit函数/模块用来训练模型参数
lr.fit(X_train, y_train)
# 使用训练好的模型lr对X_test进行预测，结果存储在遍历lr_y_predict中
lr_y_predict = lr.predict(X_test)
# 调用SGDClassifier中的fit函数用来训练模型参数
sgdc.fit(X_train, y_train)
# 使用训练好的模型sgdc对X_test进行预测，结果存储在变量sgdc_y_predict中
sgdc_y_predict = sgdc.predict(X_test)
# 使用logisticRegression模型自带的评分函数score获得模型在测试集上的准确性结果
print 'Accuracy of LR Classifier:', lr.score(X_test, y_test)
print classification_report(y_test, lr_y_predict, target_names=['Benign',
                                                                'Malignant'])
# 使用随机梯度下降模型自带的评分函数score获得模型在测试集上的准确性结果
print 'Accuracy of SGD Classifier:', sgdc.score(X_test, y_test)
print classification_report(y_test, sgdc_y_predict, target_names=['Benign',
                                                                  'Malignant'])

if __name__ == '__main__':
    print data.shape
