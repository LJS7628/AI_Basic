#일차함수 관계식 찾기

import matplotlib.pyplot as plt

x=[-3,31,-11,4,0,22,-2,-5,-25,-14]
y=[-2,32,-10,5,1,23,-1,-4,-24,-13]


plt.plot(x,y)
plt.show()

# 판다스 데이터 프레임 만들기

import pandas as pd

df = pd.DataFrame({'X':x,'Y':y})
df

df.shape

df.head()

df.tail()

X_train=df.loc[:,['X']] #모든행, X열
X_train

Y_train=df.loc[:,['Y']] #모든행, Y열
Y_train

from sklearn.linear_model import LinearRegression #사이킷런 선형회기

lr = LinearRegression() #모델링
lr.fit(X_train,Y_train) #학습

lr.coef_ , lr.intercept_ # y= wx +b = x+1 회귀계수, y절편

lr.coef_[0]

# 예측

import numpy as np

X_new = np.array(11).reshape(1,1) #shape
X_new

lr.predict(X_new)

X_test = np.arange(11,16).reshape(-1,1) #행은 알아서 계산해라 => -1
X_test

lr.predict(X_test)  #X는 무조권 2차원으로 학습할때 열의 갯수와 예측의 열 갯수 맞춰야함

lr.score(X_train,Y_train) #결정계수 (0: 안좋은모델, 1: 좋은모델)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

mean_squared_error(Y_train,lr.predict(X_train))  #mse
mean_squared_error(Y_train,lr.predict(X_train),squared=False) #rmse
mean_absolute_error(Y_train,lr.predict(X_train)) #mae