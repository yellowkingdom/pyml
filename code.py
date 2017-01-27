# coding:utf-8

# page 38 --  code13
import pandas as pd
import numpy as np

column_names = ['Sample code number','Clump Thickness','Uniformity of Cell Size',
'Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size',
'Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=column_names)

data = data.replace(to_replace='?',value=np.nan)
data = data.dropna(how='any')
print data.shape

# page 39 --  code 14
from sklearn.cross_validation import train_test_split
# 随机采样25%数据用于测试，剩下的75%用于构建训练集合
X_train,X_test,y_train,y_test = train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)
# 查验训练样本的数量和类别分布
print y_train.value_counts()
print y_test.value_counts()

# page 40 -- code 15
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

# 标准化数据，保证每个唯独的特征数据方差为1，均值为0，使得预测结果不会被某些数据过大的特征值而主导
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test  = ss.transform(X_test)

# 初始化LogisticRegression和SGDClassifier
lr = LogisticRegression()
sgdc = SGDClassifier()

# 调用 LogisticRegression中的fit函数/模块用来训练模型参数
lr.fit(X_train,y_train)
# 使用训练好的模型lr对X_test进行预测，结果存储在变量lr_y_predict中
lr_y_predict = lr.predict(X_test)
sgdc.fit(X_train,y_train)
sgdc_y_predict=sgdc.predict(X_test)
print lr_y_predict
print sgdc_y_predict

# page 42 -- code 16
from sklearn.metrics import classification_report

print 'Accuracy of LogisticRegression:',lr.score(X_test,y_test)
print classification_report(y_test,lr_y_predict,target_names=['Benign','Malignant'])
print 'Accuracy of SGDClassifier:',sgdc.score(X_test,y_test)
print classification_report(y_test,sgdc_y_predict,target_names=['Benign','Malignant'])

