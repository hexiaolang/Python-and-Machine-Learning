# -*- coding:utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# 使用加载器读取数据
iris = load_iris()
print iris.data.shape
print iris.DESCR
# 分割训练集、测试集数据
X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                    iris.target,
                                                    test_size=0.25,
                                                    random_state=33)
# 数据标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
# 使用KNN对测试样本进行分类
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_predict = knc.predict(X_test)

print 'The Accuracy of K-Nearest Neighbor Classifier is', knc.score(X_test, y_test)
print classification_report(y_test, y_predict, target_names=iris.target_names)
