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
"""样本"""
y = data[['PE']]
y.head()
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print X_train.shape
print y_train.shape
print X_test.shape
print y_test.shape

"""
结果如下：

(7176, 4)
(7176, 1)
(2392, 4)
(2392, 1)　　　　

　　　可以看到75%的样本数据被作为训练集，25%的样本被作为测试集。
"""

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

print linreg.intercept_
print linreg.coef_

"""
输出如下：

[ 447.06297099]
[[-1.97376045 -0.23229086  0.0693515  -0.15806957]]
　　　　这样我们就得到了在步骤1里面需要求得的5个值。也就是说PE和其他4个变量的关系如下：

　　　　PE=447.06297099?1.97376045?AT?0.23229086?V+0.0693515?AP?0.15806957?RHPE=447.06297099?1.97376045?AT?0.23229086?V+0.0693515?AP?0.15806957?RH　　　　


"""


#模型拟合测试集
y_pred = linreg.predict(X_test)
from sklearn import metrics
# 用scikit-learn计算MSE
print "MSE:",metrics.mean_squared_error(y_test, y_pred)
# 用scikit-learn计算RMSE
print "RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred))
复制代码
　　　　输出如下：

MSE: 20.0804012021
RMSE: 4.48111606657
　　　　得到了MSE或者RMSE，如果我们用其他方法得到了不同的系数，需要选择模型时，就用MSE小的时候对应的参数。

　　　　比如这次我们用AT， V，AP这3个列作为样本特征。不要RH， 输出仍然是PE。代码如下：

复制代码
X = data[['AT', 'V', 'AP']]
y = data[['PE']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
#模型拟合测试集
y_pred = linreg.predict(X_test)
from sklearn import metrics
# 用scikit-learn计算MSE
print "MSE:",metrics.mean_squared_error(y_test, y_pred)
# 用scikit-learn计算RMSE
print "RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred))
复制代码
　　　　　输出如下：

MSE: 23.2089074701
RMSE: 4.81756239919
　　　　可以看出，去掉RH后，模型拟合的没有加上RH的好，MSE变大了。
8. 交叉验证
　　　　我们可以通过交叉验证来持续优化模型，代码如下，我们采用10折交叉验证，即cross_val_predict中的cv参数为10：
复制代码
X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(linreg, X, y, cv=10)
# 用scikit-learn计算MSE
print "MSE:",metrics.mean_squared_error(y, predicted)
# 用scikit-learn计算RMSE
print "RMSE:",np.sqrt(metrics.mean_squared_error(y, predicted))
复制代码
 　　　　输出如下：

MSE: 20.7955974619
RMSE: 4.56021901469
　　　　可以看出，采用交叉验证模型的MSE比第6节的大，主要原因是我们这里是对所有折的样本做测试集对应的预测值的MSE，而第6节仅仅对25%的测试集做了MSE。两者的先决条件并不同。

 

9. 画图观察结果
　　　　这里画图真实值和预测值的变化关系，离中间的直线y=x直接越近的点代表预测损失越低。代码如下：

复制代码
fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()