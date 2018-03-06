# -*- coding:utf-8 -*-
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error


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

# 算数平均
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train, y_train)
uni_knr_y_predict = uni_knr.predict(X_test)

# 根据距离加权回归
dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train, y_train)
dis_knr_y_predict = dis_knr.predict(X_test)

print 'The R-squared value of uniform-weighted KNeighorRegression'\
    ,uni_knr.score(X_test, y_test)
print 'The mean squared error of uniform-weighted KNeighorRegression'\
    ,mean_squared_error(ss_y.inverse_transform(y_test),
                        ss_y.inverse_transform(uni_knr_y_predict))
print 'The mean absolute error of uniform-weighted KNeighorRegression'\
    ,mean_absolute_error(ss_y.inverse_transform(y_test),
                         ss_y.inverse_transform(uni_knr_y_predict))

print 'The R-squared value of distance-weighted KNeighorRegression'\
    ,dis_knr.score(X_test, y_test)
print 'The mean squared error of distance-weighted KNeighorRegression'\
    ,mean_squared_error(ss_y.inverse_transform(y_test),
                        ss_y.inverse_transform(dis_knr_y_predict))
print 'The mean absolute error of distance-weighted KNeighorRegression'\
    ,mean_absolute_error(ss_y.inverse_transform(y_test),
                         ss_y.inverse_transform(dis_knr_y_predict))
