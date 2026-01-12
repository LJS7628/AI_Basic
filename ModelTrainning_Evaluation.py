"""X = 독립변수(특성)

Y = 종속변수(영향을 받는 변수) = target

*   연속형 => 이 문제를 푸는 알고리즘 : 선형회귀(LinearRegression)
*   범주형 => 이 문제를 푸는 알고리즘 : 분류(Classification)

상관계수가 1에 가까우면 양의 상관관계 -1에 가까우면 음의 상관관계


선형회귀 (LinearRegression)

선형회귀의 목적은 가설로 세운 y=wx+b 에서 w,b를 아는 것이 목표이다.

w,b를 아는 단계

1. 가설 방정식을 세운다.
 이때 방정식은 가장 좋은 직선을 알아내는 것으로 점(데이터)과 직선(가설 방정식)의 거리가 가장 짧은 것으로 정한다.

 2. 방정식에 따른 비용 계산식을 구한다.
  이것을 mse(mean,squard,error) 평균제곱편차라고 부른다.

  3. 비용의 최소값을 구한다.
  
  3-1 : 최소제곱법(lr.fit())
  w의 미분계수가 0인 점과 b가 0인 점을 찾아 제곱하는 방법
  3-2 : 경사하강법
  데이터가 충분히 많을 때 w의 미분계수 함수에서 오른쪽에서 0으로 한 점씩 미분하여 미분계수가 0이 되는 지점을 찾는 방법

  선형회귀 방정식

  1. wx+b = 단순선형회귀
  2. wx1+ wx2+b = 다중선형회귀

  다중선형회귀는 매트릭스로 표현 가능
  [w1,w2]*[x1] + b  => H(x)=WX+b  
          [x2]



"""

import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url,sep="\s+",skiprows=22,header=None) #sep=\s+ 1개이상의 공백으로 구분하겠다.
data=np.hstack([raw_df.values[::2,:],raw_df.values[1::2,:2]]) #feature
target=raw_df.values[1::2,2] #Y값 (집값)

data.shape

target

df = pd.DataFrame({'CRIM':data[0],'ZN':data[1],'INDUS':data[2],'CHAS':data[3],'NOX':data[4],'RM':data[5],'AGE':data[6],'DIS':data[7],"RAD":data[8],'TAX':data[9],'B':data[10],'LSTAT':data[11]})

df = pd.DataFrame(data, columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])

df

#데이터 분할

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(df,target,test_size=0.2,random_state=0)

X_train.shape,X_test.shape,y_train.shape,y_test.shape

#모델 & 학습
from sklearn.linear_model import LinearRegression

lr = LinearRegression() # 모델 정의 y=wx+b
lr.fit(X_train,y_train) # 학습

#평가

lr.score(X_train,y_train), lr.score(X_test,y_test) # 훈련성적과 시험성적 차이가 적을 수록 좋은 모델이다.
#훈련성적이 높고 시험성적이 낮으면 오버피팅
#훈련성적이 낮고 시험성적이 낮으면 못쓰는 모델

# 회귀계수
lr.coef_

lr.intercept_ #y 절

#평가측도
#mse,rmse, mae

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print('mse: ',mean_squared_error(y_train,lr.predict(X_train)))  #mse
print('rmse: ',mean_squared_error(y_train,lr.predict(X_train),squared=False)) #rmse
print('mae: ',mean_absolute_error(y_train,lr.predict(X_train))) #mae