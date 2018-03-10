# -*- coding:utf-8 -*-
from sklearn import datasets, metrics, preprocessing,cross_validation
import skflow

# 数据预处理
boston = datasets.load_boston()
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = cross_validation.\
    train_test_split(X, y, test_size=.25, random_state=33)
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

tf_lr = skflow.TensorFlowLinearRegressor(steps=10000,
                                         learning_rate=0.01,
                                         batch_size=50)
tf_lr.fit(X_train, y_train)
tf_lr_y_pred = tf_lr.predict(X_test)

print('The mean absoluate error of Tensorflow Linear Regressor ' \
      'on boston dataset is', metrics.mean_absolute_error(
    tf_lr_y_pred, y_test))
print('The mean squared error of Tensorflow Linear Regressor' \
      'on boston dataset is', metrics.mean_squared_error(
    tf_lr_y_pred, y_test))
print('The R-squared value of Tensorflow Linear Regressor on' \
      'boston dataset is', metrics.r2_score(tf_lr_y_pred, y_test))