# coding:utf-8
# from: http://scikit-learn.org/stable/tutorial/basic/tutorial.html
from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()

from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1])  
clf.predict(digits.data[-1:])

print digits.data.shape
print digits.data[-1:].shape
print digits.data[:-1].shape