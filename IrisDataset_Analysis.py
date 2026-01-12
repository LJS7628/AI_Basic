#데이터셋 불러오기
import pandas as pd
import numpy as np

from sklearn import datasets

iris = datasets.load_iris()
iris

type(iris)

iris.keys()
#data : x 값 -> feature
#target : y값
#target_names: 0:setosa 1:versicolor 2; virginica
#feature_name : 컬럼이름 {'sepal length (cm)','sepal width (cm)','petal length (cm)', 'petal width (cm)'}

iris['data'] #nd array
iris.data #번치라는 특수한 자료구조여서 점(.)으로 접근가능

iris['target']

iris['target_names']
iris.target_names #번치라는 특수한 자료구조여서 점(.)으로 접근가능

iris['feature_names']
iris.feature_names #번치라는 특수한 자료구조여서 점(.)으로 접근가능

print(iris['DESCR'])  #y값은 class 라고도 한다.

# 데이터셋의 크기

iris.data.shape  # x값은 2차원

iris.target.shape # y값은 1차원

#앞에서 부터 7개 추출하고 싶다면?
iris.data[:7,:]

# data -> pandas dataframe 으로 바꾸기, 컬럼명
df = pd.DataFrame(iris['data'],columns=iris['feature_names'])

# 컬럼명 바꾸기
# sepal length (cm) -> sepal_length
# sepal_width
# petal_length
# petal_width

df.columns=['sepal_length','sepal_width','petal_length','petal_width']

#컬럼명 바꾸는 또다른 방법
df.rename({'sepal length (cm)':'sepal_length'},axis='columns')

# 마지막 컬럼을 'target' 이름으로 컬럼 추가 해보세요.

df['target']=iris['target']
df

#데이터 탐색(EDA)

# 데이터 프레임의 기본정보

df.info()

# 통계정보 요약 -> describe()

df.describe()

#count : null를 제외한 데이터 갯수
#mean : 평균값
#std : 편차 (값이 클수록 편차가 심하고 작을 수록 밀도가 높음)
#min :최소값
#max : 최댓값

import matplotlib.pyplot as plt

plt.boxplot(df.iloc[:,:-1]) #df 데이터의 전체를 가져와라
plt.show()

#박스 크기 : 데이터의 분포 (작으면 비슷한 값으로 구성되어있음=분산이 크다)
#중앙선 : 평균
#동그라미 : 이상치

# 결측치 확인

df.isnull()

df.isnull().sum() #axis=0 생략 0이면 값이 있음 아니면 그 수 만큼 null임
#결측값이 없음

#중복값 확인

df.duplicated()

df.duplicated().sum()

#중복값 출력

df.loc[df.duplicated(),:]

# 조건 : sepal_length == 5.8 & patal_width == 1.9 인 행을 가져오기

df.loc[(df.sepal_length==5.8) & (df.petal_width==1.9),:]

# 중복된 데이터 제거

df=df.drop_duplicates()
#df.drop_duplicates(inplace=True)

df.shape

df.loc[(df.sepal_length==5.8) & (df.petal_width==1.9),:]

# 상관분석 : 상관이 있다(양의 상관 : x값 증가 y값 증가, 음의 상관 : x값 증가 y값 감소)
# 상관계수 : -1 < r < 1
# 음수 : 음의 상관
# 양수 : 양의 상관
# 0 에 가깝다 -> 상관 없다


df.corr()  #corelation 상관

# 상관관계 시각화
# 시각화 : 히트맵

import seaborn as sns

sns.set(font_scale=1.2)

sns.heatmap(data=df.corr(),annot=True) # annot : 실제 값 화면에 나타내기
plt.show()

# target 값 확인

df.target.value_counts() #각각의 레이블의 개수

