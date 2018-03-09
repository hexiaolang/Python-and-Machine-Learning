# -*- coding:utf-8 -*-
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,\
    mean_squared_error

# 导入数据
boston = load_boston()
X = boston.data
y = boston.target

# 分割训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.25,
                                                    random_state=33)

# 数据标准化
ss_X = StandardScaler()
ss_y = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

# 训练模型并预测
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
dtr_y_predict = dtr.predict(X_test)

# 评估
print 'The R-squared value of DecisionTreeRegressor:',\
    dtr.score(X_test, y_test)
print 'The mean squared error of DecisionTreeRegressor:',\
    mean_squared_error(ss_y.inverse_transform(y_test),
                       ss_y.inverse_transform(dtr_y_predict))
print 'The mean absolute error of DecisionTreeRegressor:',\
    mean_absolute_error(ss_y.inverse_transform(y_test),
                        ss_y.inverse_transform(dtr_y_predict))


