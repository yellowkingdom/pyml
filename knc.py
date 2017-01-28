# coding:utf-8

# page 52 --  code 25
from sklearn.datasets import load_iris
iris = load_iris()
print iris.data.shape
print iris.DESCR

# page 54 --  code 26
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25,random_state=33)

# page 54 -- code 27
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
knc = KNeighborsClassifier()
knc.fit(X_train,y_train)
y_predict = knc.predict(X_test)
print y_predict

# page 55 -- code 28
print 'Accuracy of K-Nearest Neighbors Classification is',knc.score(X_test,y_test)
from sklearn.metrics import classification_report
# 使用 sklearn.metrics 里面的 classification_report 模块对预测结果做更加详细的分析
print classification_report(y_test,y_predict,target_names=iris.target_names)