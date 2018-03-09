# -*- coding:utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.grid_search import GridSearchCV
from time import time

# 读取数据
news = fetch_20newsgroups('all')
# 分割数据
X_train, X_test, y_train, y_test = train_test_split(news.data[:3000],
                                                    news.target[:3000],
                                                    train_size=.25,
                                                    random_state=33)
# 简化系统搭建流程，将文本抽取与分类器串联起来
clf = Pipeline([('vect', TfidfVectorizer(stop_words='english', analyzer=
                                         'word')),
                ('svc', SVC())])
# 2个需要试验的超参数
parameters = {'svc__gamma': np.logspace(-2, 1, 4),
              'svc__C': np.logspace(-1, 1, 3)}
gs = GridSearchCV(clf, parameters, verbose=0, refit=True, cv=3)
# 执行单线程网格搜索
t0 = time()
gs.fit(X_train, y_train)
print 'done in %.3fs' % (time()-t0)
print gs.best_params_, gs.best_score_
print gs.score(X_test, y_test)

