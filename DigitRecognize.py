# -*- coding:utf-8 -*-
# 导入手写体数字加载器
from sklearn.datasets import load_digits
from sklearn.cross_validation import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

digits = load_digits()
# 分割训练集和测试集数据
X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target,
                                                    test_size=0.25,
                                                    random_state=33)
# 数据标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
# 初始化线性假设的支持向量机分类器LinearSVC
lsvc = LinearSVC()
# 模型训练
lsvc.fit(X_train, y_train)
# 利用训练好的模型对测试样本进行预测
y_predict = lsvc.predict(X_test)
# 模型评估
print 'The Accuracy of Linear SVC is', lsvc.score(X_test, y_test)
print classification_report(y_test, y_predict,
                            target_names=digits.target_names.astype(str))