#데이터셋 불러오기
import pandas as pd
import numpy as np

from sklearn import datasets

iris = datasets.load_iris()

# data -> pandas dataframe 으로 바꾸기, 컬럼명
df = pd.DataFrame(iris['data'],columns=iris['feature_names'])

# 컬럼명 바꾸기
df.columns=['sepal_length','sepal_width','petal_length','petal_width']

#마지막 컬럼을 'target' 이름으로 컬럼 추가
df['target']=iris['target']

import matplotlib.pyplot as plt

plt.boxplot(df.iloc[:,:-1]) #이 코드 해석좀
plt.show()

# 중복된 데이터 제거
df=df.drop_duplicates()

import seaborn as sns

sns.set(font_scale=1.2)

sns.heatmap(data=df.corr(),annot=True)
plt.show()

# target 값 확인
df.target.value_counts() #각각의 레이블의 개수

## day8의 수업내용

#sepal_length 값의 분포 - hist 함수

#plt.hist(df(['sepal_length']))
plt.hist(x='sepal_length',data=df)
plt.show()

#조금 더 이쁜 히스토그램
sns.displot(x='sepal_width',kind='hist',data=df)
plt.show()

sns.displot(x='petal_length',kind='hist',data=df)
plt.show()

sns.displot(x='petal_width',kind='hist',data=df)
plt.show()

#kde : 커널 밀도 함수
sns.displot(x='petal_width',kind='kde',data=df)
plt.show()

sns.displot(x='sepal_length',hue='target',kind='kde',data=df) #hue 타겟에 대해 카테고리별로 구분해서 그린다
plt.show()

sns.displot(x='sepal_width',hue='target',kind='kde',data=df)
plt.show()

sns.displot(x='petal_length',hue='target',kind='kde',data=df)
plt.show()

sns.displot(x='petal_width',hue='target',kind='kde',data=df)
plt.show()

# 나머지 3개의 피쳐 데이터(x)를 한번에 그래프로 출력
for col in ['sepal_length','sepal_width','petal_length','petal_width'] :
  sns.displot(x=col,hue='target',kind='kde',data=df)
plt.show()

#두 변수와의 관계
sns.pairplot(df,hue='target',size=2.0,diag_kind='kde')
plt.show()

# Y 타겟값 -> 연속형 : 회귀, 범주형 : 분류
# 데이터셋 분리
from sklearn.model_selection import train_test_split

x_data = df.loc[:,'sepal_length':'petal_width']
y_data = df.loc[:,'target']

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2,shuffle=True,random_state=20) #shuffle 데이터를 섞어서 선정
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

print(y_train.value_counts(),y_test.value_counts())

from sklearn.neighbors import KNeighborsClassifier

knn= KNeighborsClassifier(n_neighbors=7) #모델링 # n_neighbors의 기본값 : 5
# 하이퍼파라미터 : 사람이 인위적으로 주며 결과에 영향을 주는 중요한 변수
knn.fit(x_train,y_train) #학습

y_knn_pred = knn.predict(x_test) #예측
print("예측값 : ",y_knn_pred[:5])
print("정답 : ",y_test.values[:5])

from sklearn.metrics import accuracy_score

print(knn.score(x_train,y_train),knn.score(x_test,y_test)) # 정확도, accuracy
print(sum(knn.predict(x_test)==y_test)/30) #정확도

knn_acc = accuracy_score(y_test,y_knn_pred) #정확도 평가
print("Accuracy : %.4f"% knn_acc)

#from sklearn.metrics import precision_score, recall_score,accuracy_score

#precision_score(y_test,knn.predict(x_test))
#recall_score(y_test,knn.predict(x_test))
#accuracy.score(y_test,knn.predict(x_test))

import matplotlib.pyplot as plt

k_range = range(1,11)
k_range = list(k_range)
train_score = [] #train_accuracy를 담을 리스트
test_score = [] #test_accuracy를 담을 리스트

for i in k_range :
  model = KNeighborsClassifier(n_neighbors=i).fit(x_train,y_train) #모델링+학습
  train_score.append(model.score(x_train,y_train))
  test_score.append(model.score(x_test,y_test))

#시각화

# x축 : k값 1~10
# y축 : train_score

# x축 : k값 1~10
# y축 : test_score

# 겹쳐진 그래프 plot, legend 라인그래프
plt.plot(k_range,train_score,label='Train Accuracy')
plt.plot(k_range,test_score,label='Test Accuracy')
plt.xticks(k_range) # x축 눈금
plt.title('Find Best K-Value in iris') # 그래프 제목
plt.legend() # 범례 추가
plt.grid() # 눈금자 추가
plt.show()

# 연습문제

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
cancer #0은 정상 1은 암환자

cancer.data
cancer.feature_names

#나의 답안
# x,y 정의
x=cancer['data']
y=cancer['target']

# 데이터 분할
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# 모델링/학습
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)

# 평가
print(knn.score(X_train,y_train),knn.score(X_test,y_test))
# 최적의 k값
train_score = []
test_score = []
k_range = range(1,11)
for i in k_range :
  knn = KNeighborsClassifier(n_neighbors=i).fit(X_train,y_train)
  train_score.append(knn.score(X_train,y_train))
  test_score.append(knn.score(X_test,y_test))


plt.plot(k_range,train_score,label='Train Accuracy')
plt.plot(k_range,test_score,label='Test Accuracy')
plt.xticks(k_range) # x축 눈금
plt.title('Find Best K-Value in cancer') # 그래프 제목
plt.legend() # 범례 추가
plt.grid() # 눈금자 추가
plt.show()

# 교수님 답안

# x,y 정의
df = pd.DataFrame(cancer.data,columns=cancer.feature_names)
# 데이터 분할
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,
                                                 stratify=cancer.target, #비율을 맞춰준다.
                                                 random_state=0)
np.bincount(cancer.target) #데이터 확인
# 모델링/학습
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
# 평가
knn.score(X_train,y_train),knn.score(X_test,y_test)
# 최적의 k값
train_score = []
test_score = []
k_range = range(1,11)
for i in k_range :
  knn = KNeighborsClassifier(n_neighbors=i).fit(X_train,y_train)
  train_score.append(knn.score(X_train,y_train))
  test_score.append(knn.score(X_test,y_test))

np.argmin(np.array(train_score)-np.array(test_score))

#시각화

plt.plot(k_range,train_score,label='Train Accuracy')
plt.plot(k_range,test_score,label='Test Accuracy')
plt.xticks(k_range) # x축 눈금
plt.title('Find Best K-Value in cancer') # 그래프 제목
plt.legend() # 범례 추가
plt.grid() # 눈금자 추가
plt.show()

#k = 7 일 때가 가장 성능이 좋다.

knn = KNeighborsClassifier(n_neighbors=7).fit(X_train,y_train)

## SVM : 서포트 벡터 머신(분류)

from sklearn.svm import SVC

svc = SVC() # 정의 - 모델링
svc.fit(X_train,y_train) # 학습

svc.score(X_train,y_train), svc.score(X_test,y_test) #평가

# day 10

from sklearn.linear_model import LogisticRegression

lrc = LogisticRegression(max_iter=10000)
lrc.fit(X_train,y_train)

#평가

lrc.score(X_train,y_train),lrc.score(X_test,y_test)

#연습문제
# x_test 에 대해서 lrc 모델로 예측값을 구하시오

lrc.predict(X_test)[:5]

y_test[:5]

# 확률값 예측
# 위스콘신대학의 암데이터
# 0 : 정상 1 : 암
lrc.predict_proba(X_test)[:5]
# 0.012 , 0.98   => 1이 될 확률이 높다.
# 0.99 , 0.000016 = > 0이 될 확률이 높다.

#연습문제
# iris 데이터에 대해서 logistic regression으로 모델링하고,
# 평가, 확률값도 출력해보시오

from sklearn import datasets

iris = datasets.load_iris()

df = pd.DataFrame(iris['data'],columns=iris['feature_names'])

df.columns=['sepal_length','sepal_width','petal_length','petal_width']

df['target']=iris['target']


x_data = df.loc[:,'sepal_length':'petal_width']
y_data = df.loc[:,'target']

X_train,X_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2,shuffle=True,random_state=20)

from sklearn.linear_model import LogisticRegression

lrc = LogisticRegression(max_iter=10000)
lrc.fit(X_train,y_train)

print(f'X_train_accuracy : {lrc.score(X_train,y_train)}, X_test_accuracy : {lrc.score(X_test,y_test)}')
print(f'predicttion 결과 : {lrc.predict(X_test)[:5]}')
print(f'prediction proba 결과 : {lrc.predict_proba(X_test)[:5]}')

np.argmax(lrc.predict_proba(X_test),axis=1)

#의사결정나무

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(max_depth=3, random_state=20)
dtc.fit(X_train, y_train)

# 평가
dtc.score(X_train, y_train), dtc.score(X_test, y_test)

# 예측 (X_test 예측)
dtc.predict(X_test)

#정답
y_test.values

!pip install graphviz

from sklearn.tree import export_graphviz
export_graphviz(dtc,  out_file='model.dot',
                feature_names=iris['feature_names'],
                class_names=iris['target_names'], impurity=True, filled=True)

import graphviz
with open('model.dot') as f:
    data = f.read()
graphviz.Source(data)

# 연습문제
# 암데이터로 의사결정나무 알고리즘을 만들고
# 평가, 예측
# 시각화



from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer['data'],
                                                    cancer['target'],
                                                    stratify=cancer['target'],
                                                    random_state=0)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(x_train, y_train)
print(f'Train accuracy: { model.score(x_train, y_train)}   Test accurary:{model.score(x_test, y_test)}')

export_graphviz(model,  out_file='model3.dot',
                feature_names=cancer['feature_names'],
                class_names=cancer['target_names'], impurity=True, filled=True)

with open('model3.dot') as f:
    data = f.read()
graphviz.Source(data)

from sklearn.neighbors import KNeighborsClassifier

knn= KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train) #학습

from sklearn.svm import SVC

svc = SVC() # 정의 - 모델링
svc.fit(x_train,y_train) # 학습

#Voting

from sklearn.ensemble import VotingClassifier #분류

hvc = VotingClassifier(estimators = [('KNN',knn),('SVM',svc),('DT',dtc)],voting='hard')
hvc.fit(x_train,y_train)

hvc.score(x_train,y_train), hvc.score(x_test, y_test)

knn.score(x_test,y_test)

svc.score(x_test,y_test)

model.score(x_test,y_test)

## 연습문제
## x_test 앞에서 5개 데이터를 예측해보자

hvc.predict(x_test)[:5]
hvc.predict(x_test[:5])

y_test[:5]

## Bagging(랜덤 포레스트)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=50,max_depth=3,random_state=20)
rfc.fit(x_train,y_train)

#정확도
rfc.score(x_train,y_train),rfc.score(x_test,y_test)

## 부스팅

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=50,max_depth=3,random_state=20)
gbc.fit(x_train,y_train)

gbc.score(x_train,y_train),gbc.score(x_test,y_test)

!pip install xgboost

from xgboost import XGBClassifier

xgbc = XGBClassifier(n_estimators=50,max_depth=3,random_state=20)
xgbc.fit(x_train,y_train)

xgbc.score(x_train,y_train),xgbc.score(x_test,y_test)

#교차검증

X_tr, X_val,y_tr,y_val = train_test_split(x_train,y_train,test_size=0.3,random_state=20)
X_tr.shape,X_val.shape

rfc = RandomForestClassifier(n_estimators=50,max_depth=3,random_state=20)
rfc.fit(X_tr,y_tr)

#검증
rfc.score(X_tr,y_tr),rfc.score(X_val,y_val)

rfc.score(x_test,y_test)

### KFold 교차 검증

from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=20)

for idx, (tr_idx, val_idx) in enumerate(kfold.split(x_train), 1):
  print(f'fold:{idx}--------------------------------')
  print(f'train: {len(tr_idx)},  {tr_idx[:10]}')
  print(f'validation:{len(val_idx)}, {val_idx[:10]}')

# 훈련용 데이터, 검증용데이터를 이용해서 폴드별로 학습 후 결과
val_score = []

for idx, (tr_idx, val_idx) in enumerate(kfold.split(x_train), 1):
  print(f'fold:{idx}--------------------------------')

  X_tr, X_val = x_train[tr_idx, :], x_train[val_idx,:]
  y_tr, y_val = y_train[tr_idx], y_train[val_idx]

  #학습
  rfc = RandomForestClassifier(max_depth=5,random_state=20)
  rfc.fit(X_tr,y_tr)
  s = rfc.score(X_val,y_val)
  val_score.append(s)
  print(f'{s:.4f}')

np.mean(val_score)

