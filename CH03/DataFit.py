# -*- coding:utf-8 -*-
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

regressor = LinearRegression()
regressor.fit(X_train, y_train)

xx = np.linspace(0, 26, 100)
xx = xx.reshape(xx.shape[0], 1)
yy = regressor.predict(xx)

# 对回归预测直线作图
plt.scatter(X_train, y_train)
plt1, = plt.plot(xx, yy, label="Degree=1")
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1])
plt.show()

print 'The R-squared value of Linear Regerssor performing on the' \
      'training data is', regressor.score(X_train, y_train)

# 多项式特征产生器
# 映射出2次多项式特征
ploy2 = PolynomialFeatures(degree=2)
X_train_ploy2 = ploy2.fit_transform(X_train)

# 线性模型
regressor_ploy2 = LinearRegression()
regressor_ploy2.fit(X_train_ploy2, y_train)

xx_ploy2 = ploy2.transform(xx)
yy_ploy2 = regressor_ploy2.predict(xx_ploy2)

plt.scatter(X_train, y_train)
plt1, = plt.plot(xx, yy, label="Degree=1")
plt2, = plt.plot(xx, yy_ploy2, label="Degree=2")
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1, plt2])
plt.show()

print 'The R-squared value of Polynominal Regerssor(Degree = 2) performing on the' \
      'training data is', regressor_ploy2.score(X_train_ploy2,
                                                y_train)

ploy4 = PolynomialFeatures(degree=4)
X_train_ploy4 = ploy4.fit_transform(X_train)

# 线性模型
regressor_ploy4 = LinearRegression()
regressor_ploy4.fit(X_train_ploy4, y_train)

xx_ploy4 = ploy4.transform(xx)
yy_ploy4 = regressor_ploy4.predict(xx_ploy4)

plt.scatter(X_train, y_train)
plt1, = plt.plot(xx, yy, label="Degree=1")
plt2, = plt.plot(xx, yy_ploy2, label="Degree=2")
plt3, = plt.plot(xx, yy_ploy4, label="Degree=4")
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1, plt2, plt3])
plt.show()

print 'The R-squared value of Polynominal Regerssor(Degree = 4) performing on the' \
      'training data is', regressor_ploy4.score(X_train_ploy4,
                                                y_train)

X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]
# 加入L1范数正则化
lasso_ploy4 = Lasso()
lasso_ploy4.fit(X_train_ploy4, y_train)
X_test_ploy4 = ploy4.transform(X_test)
print '测试数据评估'
print '普通4次多项式:', regressor_ploy4.score(X_test_ploy4, y_test)
print 'L1正则化多项式:', lasso_ploy4.score(X_test_ploy4, y_test)

print ''
print 'L1正则化参数列表：', lasso_ploy4.coef_
print '普通多项式参数列表：', regressor_ploy4.coef_
print '普通多项式参数差异:', np.sum(regressor_ploy4.coef_**2)

# L2正则化4次多项式
ridge_poly4 = Ridge()
ridge_poly4.fit(X_train_ploy4, y_train)
print 'L2正则化多项式:', ridge_poly4.score(X_test_ploy4, y_test)
print ridge_poly4.coef_
print np.sum(ridge_poly4.coef_**2)


