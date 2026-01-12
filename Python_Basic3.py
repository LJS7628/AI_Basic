# 예외처리
# try ~ except

data = {'name': 'Jay','age':20,'grage':[3.0,4.0,3.5,4.2]}
data

# data['address']
data.get('address')

try:
  print('주소',data['address'])
except :
  print('주소정보가 없습니다.')

#finally -> try /except /finally (무조건)

try :
  #print('주소',data['address'])
  print('주소',data['name'])
except :
  print('주소정보가 없습니다.')
finally :
  print('모든 작업이 완료가 되었습니다.')

# 함수
# def 함수명(a,b): return a+b

def test(a,b) : # 정의
  return a+b

print(test(1,2))

# cal_modulo라는 함수를 정의한다. {숫자 a,b를 입력받고 a를 b로 나눈 나머지를 출력한다.}
#재사용 성을 생각해서 input은 함수 외부에서 사용하는 것이 좋다
#함수를 구성할 때 input을 사용하지 말자
def cal_modulo(a,b) :
  return a % b

a = int(input("a를 입력하시오 : "))
b = int(input("b를 입력하시오 : "))
print(cal_modulo(a,b))

num_pairs=[(5,3),(2,2),(10,3)]

for i in num_pairs :
  print(i[0],i[1])

for i,j in num_pairs:
  # 함수를 이용해서 a,b 인자로 넘겨서 a를 b로 나눈 나머지를 출력하세요.
  print(cal_modulo(i,j))

# 숫자 쌍을 원소로 갖는 리스트(num_pair_list)를 입력 받아서, 나머지 연산을 해서 함께 출력하는 cal_paris_modulo 함수를 정의한다.

# 1) 숫자 쌍의 값들에 대한 나머지를 구해서, result 딕셔너리의 원소를 추가한다. 이때 숫자 쌍(투플)이 키가 된다.

# 2) for 반복문으로 함수가 반환하는 나머지를 modulo 함수에 입력한다. 함수가 변환하는 값을 mod_pairs 변수에 저장하고 출력한다.

# num_pairs 리스트를 cal_paris_modulo 함수에 입력한다. 함수가 반환하는 값을 mod_pairs변수에 저장하고 출력한다.

num_pairs = [(5,3),(2,2),(10,3)]

def cal_pairs_modulo(num_pair_list) :
  result = {}
  for a , b in num_pair_list:
    modulo = cal_modulo(a,b)
    result[(a,b)] = modulo  # 딕셔너리에서의 키는 변경 안되는
  return result

mod_pairs = cal_pairs_modulo(num_pairs)
mod_pairs

def print_mod_pairs():
  print(cal_pairs_modulo(num_pairs))

print_mod_pairs()
print(print_mod_pairs())

#지역변수, 로컬변수

result

#전역변수, global 변수

num_pairs

#add_one 함수 정의 (숫자를 입력받아서 1을 더한 값을 반환)

def add_one(a) :
  return a+1

print(add_one(2))

[1,2,3]
# add_one 이용 -> 결과[2,3,4]

add_one_list = []
for i in [1,2,3] :
  add_one_list.append(add_one(i))

add_one_list

#람다함수 = 익명함수

def add_one(num) :
  return num+1

(lambda num :num +1)(1)

test = lambda num :num +1
test(1)

#람다함수로 변경
# 숫자 리스트의 원소에 각각 1을 더하기

add_one_lambda = []
add_func = lambda x : add_one_lambda.append(x+1)

for x in [1,2,3]:
  add_func(x)

add_one_lambda

# 두 숫자를 더하는 lambda 함수 만들기
# 결과 (2,3) -> 5

add_func = lambda x,y : x+y
add_func(2,3)

(lambda x,y : x+y)(2,3)

#파이썬의 내장함수

# 합산 sum

numbers = [1,2,3,4,5,6,7,8,9,10]
sum(numbers)

# 최대값 (max)

max(numbers)

# 최소값 (min)
min(numbers)

# 원소의 길이 (len)
len(numbers)

# enumerate 함수 - 열거하다
for i, num in enumerate(numbers,start=60):
  print(i,num)

# range 함수  range(start,end,step) default(1,len,1)
list(range(10))

# eval 함수

eval('print(numbers)')

# map 함수
# numbers +1  모든 인자에 1를 더하고 싶음 하지만 이 코드는 에러남

lambda x : x+1

#map(함수, 데이터)
list(map(lambda x : x+1,numbers))

# 연습문제
# numbers 에 2배
# [2,4,6 ... 20]
# 람다함수, 맵 함수를 이용해서 구현

lambda x : x*2

list(map(lambda x : x*2,numbers))

# filter 함수
#짝수만 거르기

even_num = lambda x : x%2 == 0

# filter(함수,데이터)

list(filter(even_num,numbers))

# int() 정수형으로 바꿔주는 함수
# str() 문자열로 바꿔주는 함수
# round() 반올림 함수

print(int(3.14),str(3.14),round(3.14,1))

# reversed 역순함수

list(reversed(numbers))

# sorted 함수
sorted([3,2,1,4])

# zip 함수  = 묶어주는 함

chars = ['a','b','c']
nums = [1,2,3]
dict(zip(chars,nums))