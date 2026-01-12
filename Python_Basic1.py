1+1 #shift enter

"""#의 갯수에 따라 텍스트의 크기가 작아진다. (최대6개)"""

print('hello')

var2 = 3.14

type(var2)

var3 = 3.3

type(var3)

# 연습문제 1
# var3에는 1과2를 더한 값을 var4에는 3과0.14를 더한 값을 type으로 확인한다.

var3 = 1 +2
type(var3)

var4 = 3+0.14
type(var4)

str1 = 'easy'

#연습문제 2
#"파이썬 딥러닝" 문자열을 str2 변수에 입력한다.
#str2 변수의 자료형을 확인한다. type 함수의 리턴 값을 print함수로 출력한다.

str2 = "파이썬 딥러닝"
print(str2)

len(str2)

print(str1[0],str1[1],str1[2],str1[3])

print(str1[-4],str1[-3],str1[-2],str1[-1])

str1[::]

str1[1::2]

# 연습문제 3
#str1 변수의 값인 "Easy" 문자열 뒤에서 3번째 부터 1번째 문자까지
# 슬라이싱 방법으로 추출한다.

str1[-3:-1]

str1[1:]

str1[:1]

# 연습문제 4
# "easy" 문자열 뒤에서 3번째 문자부터 끝까지

str1[-3:]

str4 = "%s 파이썬 딥러닝" %str1
str4 = str4+str1

str4

# f-string

a = '김'
f'안녕하세요. {a} 입니다.'

i = 1
j= 2
print(f'{i}+{j}={i+j}')

#List

list1 = [1,2,3,4,5]
list1

type(list1)

# 숫자 10을 원소로 갖는 리스트를 만들고, list2 변수에 저장한다.
#list2 변수의 값을 print 함수로 출력한다.
#list2 변수의 자료형을 type 함수로 확인하고, print 함수로 출력한다.

list2 =[10]
print(list2)
print(type(list2),list2)

#빈리스트

list3 = []
list3 = list() # 함수
list3

# 리스트 3개(list1, list2, list3)를 원소로 갖는 리스트를 생성하고
# list_of_list라는 변수에 저장한다.  // 2차원 리스트
# list_of_list 변수이름을 입력하여 데이터를 확인

list_of_list = [list1,list2,list3]
list_of_list

type(list_of_list)

# list_of_list 변수의 1번째 원소를 인덱싱 추출
list_of_list[0]

list_of_list[list1[0]]

list_of_list[0][1]

list_of_list[0][-1]

# list_of_list 변수의 0번째부터 1번째 원소를 범위를 지정하여 슬라이싱 추출

list_of_list[0:2]
list_of_list[:2]

# list1 변수는 리스트 원소로 갖는다. 숫자 100을 새로운 원소로 추가한다. (append 활용)
# list1 변수를 출력한다.

list1.append(100)

list1

#list3 변수는 리스트를 원소로 갖는다.
# 리스트에 숫자 7,8,9를 순서대로 새로운 원소로 추가한다. (append 활용)
# list3 변수를 출력한다.

list3.append(7)
list3.append(8)
list3.append(9)
list3.append('a')
list3

list_of_list[0].append(100)
list_of_list

# 튜플

tuple1 = (1,2)
print(type(tuple1),tuple1)

#문자열 "a", "b", 숫자 100을 원소로 갖는 투플(tuple)을 만들고,
# tuple2 이라는 변수에 저장한다.
# tuple2 변수를 출력한다.

tuple2=('a','b',100)
print(tuple2) # 두개 이상의 자료형이 들어간다.

#빈튜플
tuple3 = ()
tuple3 = tuple()
tuple3

list3.append('a')
list3

tuple2[0],tuple2[1],tuple2[2],tuple2[-1]

#튜플 슬라이싱
# tuple2 변수에 저장된 튜플의 2번째 원소부터 마지막 원소까지 범위

tuple2[1:]

tuple2[0]='aa'

list1[-1]=10
list1

t1 = (1,2,3,4)
l1 = [1,2,3,4]

import sys

sys.getsizeof(t1), sys.getsizeof(l1)

#딕셔너리

dict1 = {'name':'Jay','age':20}
dict1

type(dict1)

dict1['name']

dict1['age']

# 나이를 20 -> 21 로 변경
# key 하나에 여러개의 value를 가질 수 있다. 만약 value가 list라면 가능하며 list의 인덱싱, 슬라이싱도 가능하다.
dict1['age'] = 21

# 겹치는 키가 없으면 추가된다.
# 키 삭제 del dict1['age1']

dict1['age1'] =21
dict1

# dict1 딕셔너리에 새로운 원소를 추가한다.
# 'grade' 키에 매칭되는 값으로 리스트([3.0,4.0,3.5,4.2])를 추가한다.
# dict1 변수를 출력한다.

dict1['grade']=[3.0,4.0,3.5,4.2]
dict1

dict1['grade']

dict1.keys()

dict1.values()

dict1.items()

dict1.get('name') #key에 따른 value 반환

#dict1['age1']  == error
dict1.get('age1')
dict1.get('age2','값이 없습니다.')

# 산술연산자(+,-,*,/)

1+2
new_str = "Good" +" "+"Morinng"
new_str

# "good" + 3  == error
"good" + str(3)

new_str *3

2-3
3*4
5/3
5//3 # 몫

#논리연산자(T F && ||)

True

not True

type(False)

#AND

print(True and True)
print(True and False)
print(False and False)

#OR

print(True or True)
print(True or False)
print(False or False)

#비교연산자(< > = !=)

3 == 3

3 != 3

'a' == 'a'

'a' != 'a'

# "4가 5보다 작다" 라는 명제를 print 함수로 출력한다.
# "4가 5보다 작거나 같다" 라는 명제를 print 함수로 출력한다.
# "4가 5보다 크거나 같다" 라는 명제를 print 함수로 출력한다.
# "4가 5보다 크다" 라는 명제를 print 함수로 출력한다.


print(4<5)
print(4<=5)
print(4>=5)
print(4>5)

# "4가 5보다 작다" 와 "4가 5보다 작거나 같다"를 AND 연산 처리한다.

4<5 and 4<=5

## "4가 5보다 작다" 와 "4가 5보다 크다"를 OR 연산 처리한다.
4 <5 or 4>5

#"문자열 'a'가 'a''와 다르다" 라는 명제를 부정한다. (not 명령)

not 'a' !='a'

### 제어문

# if 구문

a = 4
if a > 2 :
  print('크다')
else :
  print('작다')

a % 2 == 0

if a % 2 == 0 :
  print('짝수')
else :
  print('홀수')

if a> 5:
  print('5보다 크다')
elif a> 3 :
  print('3보다 크다')
else :
  print('3보다 작다')

# c = -1
# if 조건식 : c 변수의 값이 0보다 크다.
# elif 조건식 : c 변수의 값이 0보다 작다.
# else 조건식 : 없음 (c 변수의 값이 0이다)

c = -1
if c> 0:
  print('0보다 크다')
elif c< 0 :
  print('0보다 작다')
else :
  print('0이다')

c = input('숫자를 입력하세요')
c

c = int(input('숫자를 입력하세요'))

if c > 0:
  print('0보다 크다')
elif c < 0 :
  print('0보다 작다')
else :
  print('0이다')

type(c) # 사용자 입력은 항상 str이다.

# for 반복문

num_list = [2,4,6]
for num in num_list :
  print(num)

#for 반복문 : 리스트의 원소를 반복하여 2의 배수와 함께 출력한다.
for num in num_list :
  print(num*2)

for num in [1,2,3]:
  print("기존: ",num)
  print("2배: ",num*2)
  print("\n")

for num in [1,2,3]:
  print(f"기존:{num}, 2배:{num*2}")

# for 반복문 ; 리스트의 원소를 반복해서 2의 배수를 계산하고,
# 빈 리스트(double)에 2의 배수를 원소로 추가한다. append

double = list()
for num in [1,2,3] :
  double.append(num*2)

double

#list comprehension

[x*2 for x in [1,2,3]]

