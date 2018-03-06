# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 从互联网上收集泰坦尼克号乘客数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
print titanic.head()

X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
# 补充age记录
X['age'].fillna(X['age'].mean(), inplace=True)
# 分割训练集、测试集数据
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=33)
# 特征转换器进行特征抽取
vec = DictVectorizer(sparse=False)
# 类别型的特征单独剥离，独立成一列特征，数值型保持不变
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))
# 使用默认配置初始化决策树分类器
dtc = DecisionTreeClassifier()
# 训练决策树模型
dtc.fit(X_train, y_train)
# 对测试样本进行预测
y_predict = dtc.predict(X_test)

print dtc.score(X_test, y_test)
print classification_report(y_predict, y_test, target_names=['died',
                                                             'survived'])
