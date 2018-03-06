# -*- coding:utf-8 -*-
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# 导入波士顿房价数据
boston = load_boston()
# print boston.DESCR

X = boston.data
y = boston.target
# 分割训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=33)

# 数据标准化
ss_X = StandardScaler()
ss_y = StandardScaler()

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

# 使用默认配置初始化线性回归器LinearRegression
lr = LinearRegression()
# 使用训练集进行建模
lr.fit(X_train, y_train)
# 对测试集进行预测
lr_y_pred = lr.predict(X_test)

# 使用默认配置初始化线性回归器SGDRegressor
sgdr = SGDRegressor()
# 使用训练集进行建模
sgdr.fit(X_train, y_train)
# 对测试集进行预测
sgdr_y_predict = sgdr.predict(X_test)

# 评价模型
print 'The value of default measurement of LinearRegression is', lr.score(X_test, y_test)
print 'The value of R-squared of LinearRegression is', r2_score(y_test, lr_y_pred)
print 'The mean squared error of LinearRegression is', mean_squared_error(ss_y.inverse_transform(y_test),
                                                                          ss_y.inverse_transform(lr_y_pred))
print 'The mean absolute error of LinearRegression is', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                            ss_y.inverse_transform(lr_y_pred))

print 'The value of default measurement of SGDRegression is', sgdr.score(X_test, y_test)
print 'The value of R-squared of SGDRegression is', r2_score(y_test, sgdr_y_predict)
print 'The mean squared error of SGDRegression is', mean_squared_error(ss_y.inverse_transform(y_test),
                                                                       ss_y.inverse_transform(sgdr_y_predict))
print 'The mean absolute error of SGDRegression is', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                         ss_y.inverse_transform(sgdr_y_predict))
