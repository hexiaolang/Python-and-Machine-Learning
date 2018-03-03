# -*- coding:utf-8 -*-
# 导入手写体数字加载器
from sklearn.datasets import load_digits
from sklearn.cross_validation import  train_test_split

digits = load_digits()
# 分割训练集和测试集数据
X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target,
                                                    test_size=0.25,
                                                    random_state=33)