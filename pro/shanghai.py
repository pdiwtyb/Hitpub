import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression  
from sklearn.linear_model import RidgeCV  
from sklearn.cross_validation import train_test_split  
  
'''''#load data 
n=100 
x = np.arange(1,100,n)+np.random.randn(n) 
y = 4*x - 3 + np.random.randn(n) 
plt . figure () 
plt . plot(x, y, 'r*', label='X') 
plt . ylabel (" Y"  ) 
plt . xlabel (" X") 
plt . legend(loc="best") 
plt . tight_layout() 
plt . show() 
'''  
data = ['C:\\Users\\123\\Desktop\\weather\\2015.txt',]  
w = np. loadtxt ( data [0] , skiprows =1)  
y = w[:,7]/10  
x = w[:,10]  
plt . figure ()  
plt . plot(x,y,"b*",label="Atmospheric pressure")  
plt . ylabel (" Temperatures"  )  
plt . xlabel ("Atmospheric pressure "  )  
plt . title (' Temperatures trent chart of Shanghai in year 2015 ')  
plt . tight_layout()  
plt . legend(loc="best")  
plt . show()  
  
  
x = x.reshape(-1, 1)  
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)  
lr = LinearRegression()  
lr . fit ( x_train , y_train)  
y_lr = lr.predict ( x_test )  
cv = RidgeCV(alphas=np.logspace(-3, 2, 100))  
cv . fit ( x_train , y_train)  
y_cv = cv.predict ( x_test )
"""
print lr.coef_  
print lr.intercept_  
print "mes of Linear Regresion squares is", np. mean(( y_lr - y_test ) ** 2)  
print "accuracy of Linear regression is",lr.score(x_test,y_test)  
print cv.coef_  
print cv.intercept_  
print "mes of Linear Regresion+Ridge squares is", np. mean(( y_cv - y_test ) ** 2)  
print "accuracy of Linear regression is",cv.score(x_test,y_test)  
  """
x1 = np.arange(len(x_test))  
plt.plot(x1,y_test,"y*-",label="Test")  
plt.plot(x1,y_lr,"ro-",label="Predict")  
plt.plot(x1,y_cv,"b^-",label="Predict+Ridge")  
plt . ylabel (" Temperatures"  )  
plt . xlabel (" Atmospheric pressure")  
plt . title (' Predict chart ')  
plt . legend(loc="best")  
plt . tight_layout()  
plt . show()  
