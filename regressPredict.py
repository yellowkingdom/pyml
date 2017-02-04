# coding:utf-8

# page 65 -- code 34
from sklearn.datasets import load_boston
boston = load_boston()
# print boston.DESCR

# page 66 -- code 35
from sklearn.cross_validation import train_test_split
import numpy as np
X=boston.data
y=boston.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)

print 'the max target value',np.max(boston.target)
print 'the min target value',np.min(boston.target)
print 'the average target value',np.mean(boston.target)

# page 67 -- code 36
# 标准化
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
ss_y = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
# y_train = ss_y.fit_transform(y_train.reshape(-1,1))
# y_test = ss_y.transform(y_test.reshape(-1,1))
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

# page 67 -- code 37
# 线性回归器预测
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
lr_y_predict = lr.predict(X_test)
# SGD回归器预测
from sklearn.linear_model import SGDRegressor
sgdr = SGDRegressor()
sgdr.fit(X_train,y_train)
sgdr_y_predict = sgdr.predict(X_test)

# page 69 -- code 38
# code 38 part 1
# 性能评测
print 'the value of default measurement of LinearRegression is',lr.score(X_test,y_test)
# 导入R2_score mean_squared_error 以及 mean_absoluate_error 用于回归性能评估
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print 'the value of R-squared of LinearRegression is',r2_score(y_test,lr_y_predict)
print 'the value of mean_squared_error of LinearRegression is',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict))
print 'the value of mean_absolute_error of LinearRegression is',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict))

# code 38 part 2
print 'The value of default measurement of SGDRegressor is ',sgdr.score(X_test,y_test)
print 'The value of R-squared of SGDRegressor is ',r2_score(y_test,sgdr_y_predict)
print 'The value of mean squared error of SGDRegressor is ',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict))
print 'The value of mean absolute error of SGDRegressor is ',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict))