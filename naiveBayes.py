# coding:utf-8

# 每个案例顺序都是 1.获取数据 2.分割数据(测试和训练) 3.预测数据 4.性能评估
# 

# page 49 --  code 21
from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset='all')
print len(news.data)
print news.data[0]

# page 49 --  code 22
from sklearn.cross_validation import train_test_split
# 随机按照25%切分测试数据和训练样本
X_train,X_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=33)

# page 50 --  code 23
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
y_predict = mnb.predict(X_test)
# print y_predict

# page 50 --  code 24
from sklearn.metrics import classification_report
print 'Accuracy of Naive Bayes Classification is',mnb.score(X_test,y_test)
from sklearn.metrics import classification_report
# 使用 sklearn.metrics 里面的 classification_report 模块对预测结果做更加详细的分析
print classification_report(y_test,y_predict,target_names=news.target_names)