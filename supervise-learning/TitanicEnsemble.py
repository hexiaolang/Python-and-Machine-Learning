# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# 从互联网上收集数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# 人工选取pclass、age以及sex作为判别乘客是否能够生还的特征
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
# 对于缺失的age数据，采用平均年龄代替
X['age'].fillna(X['age'].mean(), inplace=True)
# 分割训练集、测试集数据
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=33)
# 转化类别型特征称为特征向量
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

# 使用单一决策树进行模型训练及预测
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_y_pred = dtc.predict(X_test)

# 使用随机森林进行模型训练及预测
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_y_pred = rfc.predict(X_test)

# 使用梯度提升决策树进行模型训练及预测
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_y_pred = gbc.predict(X_test)

print 'The accuracy of decision tree is', dtc.score(X_test, y_test)
print classification_report(dtc_y_pred, y_test)

print 'The accuracy of random forest classifier is', rfc.score(X_test, y_test)
print classification_report(rfc_y_pred, y_test)

print 'The accuracy of gradient tree boosting is', gbc.score(X_test, y_test)
print classification_report(gbc_y_pred, y_test)
