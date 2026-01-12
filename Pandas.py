"""
객체지향(Object-Oriented program)

-데이터(DB) + 기능(구조적)

데이터는 변수, 기능은 함수의 형태를 띤다.
Class는 object(개체)라고도 한다.

판다스 자료구조
1. series-1차원 자료구조
-인덱스를 문자,날짜등 다양하게 지정할 수 있다.

pd.series([5,3,1,4])
s[Idx]=s[str]=s[date]
s[0]=s[A]=s[2023.10.10]

2. dataframe-2차원 자료구조
-인덱스를 문자,날짜등 다양하게 지정할 수 있다.
-인덱스,컬럼,벨류로 나눠져있다.
pd.DataFrame([[]])
"""

### add 함수 (두 숫자를 더함)

def add(num1, num2) :
  return num1 + num2

add(1,2)

# 객체이용

#cal = Calculator(1,2)
#cal.함수이름()
#cal.변수명

class Calculator:

  def __init__(self,num1,num2): #생성자
    self.num1 = num1
    self.num2 = num2
    self.result = 0

  def add(self):
    self.result = self.num1+self.num2
    return self.result

  #연습문제 사칙연산
  def substract(self):
    self.result = self.num1-self.num2
    return self.result

  def multiply(self):
    self.result = self.num1*self.num2
    return self.result

  def divide(self):
    self.result = self.num1/self.num2
    return self.result

  def change(self):
    self.num1, self.num2 = self.num2,self.num1
    print(self.num1,self.num2)

  def change2(self,num1,num2) :
    num1, num2 = num2,num1
    print(num1,num2)

cal = Calculator(1,2)
cal.num1, cal.num2,cal.result  #(1,2,0)
cal.add()
cal.result  # (3)

cal.substract()
cal.result #(-1)

print(cal.multiply())
print(cal.divide())
cal.change()

cal2 = Calculator(1,2)
cal2.substract()
cal2.multiply()
cal2.change2(100,200)

## 판다스 라이브러리 불러오기

import pandas as pd

pd.__version__

#문자열 원소를 갖는 1차원 배열 - 리스트

data1 = list("abcde")
data1
type(data1)

#series 만들기

s1 = pd.Series(data1)
s1 # object는 문자열
type(s1)

print(s1.values,s1.index)

s1[0]

s1.loc[:]

s1[1:3]

s1.loc[1:3]

# 튜플로 시리즈 만들기

data2 = (1,2,3.14,100,-10)
s2 = pd.Series(data2)
s2

# 시리즈를 결합하여 데이터 프레임으로 변환

dict_data = {'c0':s1,'c1':s2}
df1= pd.DataFrame(dict_data)
df1

type(df1)

pd.DataFrame([[1,2],[3,4]],index=['A','B'],columns=['col1','col2'])

pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],index=['A','B','C'],columns=['X1','X2','X3'])

df1.index

df1.columns

df1.values

df1.columns=['string','number']
df1

df1.index = ['r0','r1','r2','r3','r4'] #인덱스명 변경
df1

df1['number']  #대괄호는 데이터프레임의 열을 가져온다.

#df1.loc[행,열] 단 열은 생략 가능(모든 요소 추출)
df1.loc['r0']

df1.loc['r0','number']

# 데이터 부분 추출

df1.loc['r2':'r3']

df1.loc['r2':'r3','number']

df1.loc['r2','string':'number']
df1.loc['r2',:]
df1.loc['r2']

df1.loc[:,'string']
df1.loc['r0':'r4','string']

df1.loc['r2':'r3',:]

df1.to_csv('myfile.csv')

df2 = pd.read_csv('myfile.csv',index_col='Unnamed: 0')
df2 # 인덱스의 네임은 필요없기 때문에 Unnamed 라고 한다.

df2.columns=['col1','col2']
df2

# 데이터프레임.rename({before:after, before:after},axis='columns')
df2.rename({'col1':'X1'},axis='columns')  #뷰의 컬럼만 변경 원본은 변경안됨

df2.rename({'col1':'X1'},axis='columns', inplace=True)  #실제 원본데이터를 수정

df2

#컬럼삭제
#drop은 기본적으로 행을 삭제한다.
#axis=0 행 ->기본값, axis= 1열
df2.drop('X1',axis=1)

df2.drop('X1',axis=1,inplace=True)

df2 = pd.read_csv('myfile.csv',index_col='Unnamed: 0')
df2