# coding:utf-8

# page 45 --  code 17
from sklearn.datasets import load_digits

digits = load_digits()
print digits.data.shape

# page 45 --  code 18
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.25,random_state=33)

print y_train.shape
print y_test.shape

# page 46 --  code 19
from sklearn.preprocessing  import StandardScaler
from sklearn.svm            import LinearSVC

# 标准化处理训练和测试的特征数据
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test  = ss.transform(X_test)

# 初始化线性假设的支持向量机分类器 LinearSVC
lsvc    = LinearSVC()
# 进行模型训练
lsvc.fit(X_train,y_train)
# 利用训练好的模型对测试样本的数据类别进行预测，预测结果存储在变量 y_predict 中。
y_predict = lsvc.predict(X_test)
print y_predict

# page 47 --  code 20
print 'Accuracy of LinearSVC is',lsvc.score(X_test,y_test)
from sklearn.metrics import classification_report
# 使用 sklearn.metrics 里面的 classification_report 模块对预测结果做更加详细的分析
print classification_report(y_test,y_predict,target_names=digits.target_names.astype(str))