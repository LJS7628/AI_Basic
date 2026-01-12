"""
for문 연습문제 1(합격,불합격)
-총 5명 학생의 성적이 다음과 같을 때 시험 점수가 60점 이상이면 "합격입니다." 를 그렇지 않으면 "불합격입니다."를 출력하세요
- 사용) result= [90,35,68,44,81]
"""

result = [90,35,68,44,81]

for i in result :
  if i >=60 :
    print("합격입니다.")
  else :
    print("불합격입니다.")

j= 1

"""1번째 학생은 합격
2번째 학생은 불합격
3번째 학생은 합격
4번째 학생은 불합격
5번째 학생은 합격
"""

j=1
for i in result :
  if i >=60 :
    print(str(j)+"번"+"합격입니다.")
    j=j+1
  else :
    print(str(j)+"번"+"불합격입니다.")
    j=j+1

for n,i in enumerate(result):
  print(n,i)

for n,i in enumerate(result,start=1) :
  if i >=60 :
    print(f'{n}번째 학생은 합격')
  else :
    print(f'{n}번째 학생은 불합격')

# range : 숫자범위 생성

#list(range(10))
#list(range(1,10))
#list(range(1,10,2))
#list(range(0,10,2))
#list(range(0,10,3))

for i in range(10):
  print(i, end=' ')

# 연습
# 1 3 5 7 9 11 13 15 17 19 나오도록 출력해보기

for i in range(1,20,2):
  print(i, end=' ')

# for문과 range() 함수를 이용하여 1부터 100까지의 합을 구해 출력
# sum를 사용하지 마시요 sum()함수 와 충돌 남
# sum(range(1,101)) ==> 5050
total = 0
for i in range(1,101) :
  total += i
print(total)

test2D = [[1,2,3],[4,5,6],[7,8,9,10]]

for i in test2D :
  for j in i :
    print(j, end=' ')
  print()

# 인덱스 사용
for x in range(len(test2D)):
  #print(test2D[x])
  for y in range(len(test2D[x])):
    print(test2D[x][y], end=' ')
  print()

for n,x in enumerate(test2D) :
  for m,y in enumerate(test2D[n]):
    print(test2D[n][m], end=' ')
  print()

# for 구구단 출력

#2 4 6 ..
#3 6 9 ..

result = 0
for i in range(2,10):
  print()
  for j in range(1,10):
    result = i*j
    print(result ,end=' ')

for dan in range(2,10):
  for no in range(1,10):
    print(f'{dan*no:2d}',end=' ')
  print()

#while

num =1

while num <4 :
  print('기존:',num)
  print('2배:',num*2)
  num +=1

num = 1
double =[]

while num < 4 :
  double.append(num*2)
  num = num +1

print(double)

num =1
double =[]
while True:
  double.append(num*2)
  if len(double) == 3:
    break
  num +=1

print(double)

# 로또번호 만들기
# 1 ~ 45

num_range = []
for i in range(1,46) :
  num_range.append(i)

print(num_range)

#num_range = list(range(1,46))

import random

random.shuffle(num_range)
print(num_range)

#continue : 나머지 코드를 건너뛰고 next로 간다.

for i in range(10):
  if i ==2:
    continue
  if i == 5:
    break
  print(i)

import random
lotto=[] #6개
num_range = list(range(1,46))

while len(lotto)<6 :
  random.shuffle(num_range)
  num_selected = num_range[0]
  if num_selected in lotto:
    continue
  lotto.append(num_selected)
  print(num_selected)



print(lotto)