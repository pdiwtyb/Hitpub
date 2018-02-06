import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

data = pd.read_csv('c:\\CCPP\ccpp.csv')
data.head()
data.shape
X = data[['AT', 'V', 'AP', 'RH']]
X.head()
"""����"""
y = data[['PE']]
y.head()
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print X_train.shape
print y_train.shape
print X_test.shape
print y_test.shape

"""
������£�

(7176, 4)
(7176, 1)
(2392, 4)
(2392, 1)��������

���������Կ���75%���������ݱ���Ϊѵ������25%����������Ϊ���Լ���
"""

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

print linreg.intercept_
print linreg.coef_

"""
������£�

[ 447.06297099]
[[-1.97376045 -0.23229086  0.0693515  -0.15806957]]
���������������Ǿ͵õ����ڲ���1������Ҫ��õ�5��ֵ��Ҳ����˵PE������4�������Ĺ�ϵ���£�

��������PE=447.06297099?1.97376045?AT?0.23229086?V+0.0693515?AP?0.15806957?RHPE=447.06297099?1.97376045?AT?0.23229086?V+0.0693515?AP?0.15806957?RH��������


"""


#ģ����ϲ��Լ�
y_pred = linreg.predict(X_test)
from sklearn import metrics
# ��scikit-learn����MSE
print "MSE:",metrics.mean_squared_error(y_test, y_pred)
# ��scikit-learn����RMSE
print "RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred))
���ƴ���
��������������£�

MSE: 20.0804012021
RMSE: 4.48111606657
���������õ���MSE����RMSE��������������������õ��˲�ͬ��ϵ������Ҫѡ��ģ��ʱ������MSEС��ʱ���Ӧ�Ĳ�����

���������������������AT�� V��AP��3������Ϊ������������ҪRH�� �����Ȼ��PE���������£�

���ƴ���
X = data[['AT', 'V', 'AP']]
y = data[['PE']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
#ģ����ϲ��Լ�
y_pred = linreg.predict(X_test)
from sklearn import metrics
# ��scikit-learn����MSE
print "MSE:",metrics.mean_squared_error(y_test, y_pred)
# ��scikit-learn����RMSE
print "RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred))
���ƴ���
����������������£�

MSE: 23.2089074701
RMSE: 4.81756239919
�����������Կ�����ȥ��RH��ģ����ϵ�û�м���RH�ĺã�MSE����ˡ�
8. ������֤
�����������ǿ���ͨ��������֤�������Ż�ģ�ͣ��������£����ǲ���10�۽�����֤����cross_val_predict�е�cv����Ϊ10��
���ƴ���
X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(linreg, X, y, cv=10)
# ��scikit-learn����MSE
print "MSE:",metrics.mean_squared_error(y, predicted)
# ��scikit-learn����RMSE
print "RMSE:",np.sqrt(metrics.mean_squared_error(y, predicted))
���ƴ���
 ��������������£�

MSE: 20.7955974619
RMSE: 4.56021901469
�����������Կ��������ý�����֤ģ�͵�MSE�ȵ�6�ڵĴ���Ҫԭ�������������Ƕ������۵����������Լ���Ӧ��Ԥ��ֵ��MSE������6�ڽ�����25%�Ĳ��Լ�����MSE�����ߵ��Ⱦ���������ͬ��

 

9. ��ͼ�۲���
�����������ﻭͼ��ʵֵ��Ԥ��ֵ�ı仯��ϵ�����м��ֱ��y=xֱ��Խ���ĵ����Ԥ����ʧԽ�͡��������£�

���ƴ���
fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()