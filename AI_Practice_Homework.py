import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# group1.csv를 읽어 group1 변수에 저장하세요.
# 알맞는 알고리즘을 선택하여 모델링하고, 모델링후의 성능을 출력하세요.

#1. csv 파일 읽어오기
group1 = pd.read_csv("group1.csv")
group1.head()

#2. 정보 확인
group1.info()

#3. 결측값 확인
print(group1.isnull().sum())

#4. date 필드를 Index로
group1 = group1.set_index("Date")
group1.head()

#5. 데이터 x,y 만들기
x = group1.loc[:,'x1':]
y = group1.loc[:,'Y':]

#6. 데이터 나누기 (훈련,시험)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)

#7. 잘 나눠졌는지 확인
print(X_train.shape,X_test.shape,y_train.shape,y_train.shape)

#8. 모델링 및 학습
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_train,y_train)

#9.평가(evaluation)

print(lr.score(X_train,y_train),lr.score(X_test,y_test)) # r 결정계수

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#학습데이터에 대한 mse
print('mse: ',mean_squared_error(y_train,lr.predict(X_train)))  #mse
print('rmse: ',mean_squared_error(y_train,lr.predict(X_train),squared=False)) #rmse
print('mae: ',mean_absolute_error(y_train,lr.predict(X_train))) #mae

#회귀계수, y절편 확인
print(lr.coef_[0],lr.intercept_[0])