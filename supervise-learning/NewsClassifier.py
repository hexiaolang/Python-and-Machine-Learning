# -*- coding:utf-8 -*-
# 导入新闻数据抓取器
from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# fetch_20newsgroups 需要即时从互联网下载数据
news = fetch_20newsgroups(subset='all')
print len(news.data)
print news.data[0]
# 分割训练集、测试集数据
X_train, X_test, y_train, y_test = train_test_split(news.data,
                                                    news.target,
                                                    test_size=0.25,
                                                    random_state=33)
# 文本特征向量转化模块
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)
# 使用默认配置初始化朴素贝叶斯模型
mnb = MultinomialNB()
# 利用训练数据对模型参数进行估计
mnb.fit(X_train, y_train)
# 对测试样本进行类别预测
y_predict = mnb.predict(X_test)
print 'The accuracy of Naive Bayes Classifier is', mnb.score(X_test, y_test)
print classification_report(y_test, y_predict, target_names=news.target_names)
