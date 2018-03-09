# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn import feature_selection
from sklearn.tree import  DecisionTreeClassifier
import numpy as np
from sklearn.cross_validation import cross_val_score
import pylab as pl

# 获取数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/'
                      'pub/Main/DataSets/titanic.txt')
# 分离特征和类别
y = titanic['survived']
X = titanic.drop(['row.names', 'name', 'survived'], axis=1)

# 填充缺失数据
X['age'].fillna(X['age'].mean(), inplace=True)
X.fillna('UNKNOW', inplace=True)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.25,
                                                    random_state=33)


# 特征抽取
vec = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))


# 建立决策树模型
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print dt.score(X_test, y_test)

# 特征筛选
fs = feature_selection.SelectPercentile(feature_selection.chi2,
                                        percentile=20)
X_train_fs = fs.fit_transform(X_train, y_train)
X_test_fs = fs.transform(X_test)
dt.fit(X_train_fs, y_train)
print dt.score(X_test_fs, y_test)

# 通过交叉验证，分析性能随特征筛选比例的变化
percentiles = range(1, 100, 2)
results = []

for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2,
                                            percentile=i)
    X_train_fs = fs.fit_transform(X_train, y_train)
    scores = cross_val_score(dt, X_train_fs, y_train, cv=5)
    results = np.append(results, scores.mean())
print results

# 找到体现最佳性能的特征筛选百分比
opt = np.where(results == results.max())[0][0]
print ('Optimal number of features %d' % percentiles[opt])

pl.plot(percentiles, results)
pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
pl.show()

# 使用最佳百分比
fs = feature_selection.SelectPercentile(feature_selection.chi2,
                                        percentile= percentiles[opt])
X_train_fs = fs.fit_transform(X_train, y_train)
X_test_fs = fs.transform(X_test)
dt.fit(X_train_fs, y_train)
print dt.score(X_test_fs, y_test)