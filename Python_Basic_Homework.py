"""
### 1번문제
- 1. mylist = [[1,2,3,4,5]]에 77을 추가하여 mylist가 [[1,2,3,4,5,77]]이 되도록 구현해 보자
  - append와 insert 2가지 방법을 활용해서 구현해 보자
"""

# code here

#tt = 77
mylist = [[1,2,3,4,5]]
mylist[0].append(77)  # == mylist[0].append(tt)
print(mylist)

# code here

mylist = [[1,2,3,4,5]]
mylist[0].insert(5,77) # == insert(5,tt)
print(mylist)

"""## 2번문제
- 2. 사전(Dict)를 이용하여 학생들의 이름과 점수를 입력하고 추가, 삭제, 업데이트 작업을 진행해 보자
  - score라는 dict에 철수 95점, 영희 88점, 길동 92점을 입력하자
  - score에서 수지 98점을 추가하자
  - score에서 길동 92점을 삭제하자
  - score에 민영 85점, 두리 55점, 남일 69점을 한번에 추가에 보자

"""

# code here
score = {}
for i in [1,2,3] :
  key = input("이름 : ")
  value = int(input("점수 : "))
  score[key] = value
print(score,type(score))

# score = dict([('철수',95),('영희',88),('길동',92)])
# print(score,type(score))

# code here
score['수지']= 98
print(score)

# code here
del score['길동']
print(score)

# code here
score.update({'민영':85, '두리':55,'남일':69})
print(score)

"""## 3번문제
- 3. 다음 리스트에서 중복된 항목을 없애보자(Set 타입의 속성을 이용하자)
  - [1,2,3,5,3,2,9,8,11,7,5,1]

"""

# code here

ls = [1,2,3,5,3,2,9,8,11,7,5,1]
print(ls)

# code here

set1 = set(ls)
print(set1)

"""## 4번문제
- 4. 세 문자열에서 공통된 문자만 출력해 보자(Set의 속성 이용)
  - "Python is simple",   "apple is delicious",  "programming"

"""

# code here

set_1 = set("python is simple")
set_2 = set("apple is delicious")
set_3 = set("programming")
print(set_1 & set_2 & set_3)

# a = 'Python is simple'
# b = 'apple is delicious'
# c = 'programming'
# print(set(a)& set(b) & set(c))

"""## 5번문제


- 5. 다음과 같은 규칙을 같은 숫자의 합을 for 문을 이용하여 구현하세요
  - •	1 + (1+2) + (1+2+3) + (1+2+3+4) .......(1+2+3+.....10)

"""

# code here
total = 0

for i in range(1,11) :
  for j in range(1,i+1): #range(i+i)  - 0,1,2 ...
    total += j

print(total)

"""## 6번 문제
- 6. 1 부터 100까지의 숫자중 임의의 숫자를 발생시켜 해당 숫자를 맞쳐나가는 프로그램을 구현해 보자 (NumberGuess.py)

  - random.randrange() 함수를 이용하여 1부터 100사이의 값을 발생시킨다
  - 사용자가 임의의 숫자를 입력한다.
  - 예를 들어 컴퓨터는 난수를 66을 발생시켰고 사용자가 55를 입력했으면
  - 숫자가 작습니다. 더 큰 숫자를 입력하세요 라고 출력합니다.
  - 컴퓨터가 발생시킨 숫자를 맞히면 정답은 XX  입니다. X 번만에 맞히셨습니다. 라고 출력합니다

"""

# code here
import random

rd = random.randrange(1,101)
cnt = 0
print(rd)
while True :
  user = int(input())
  if user > rd :
    cnt +=1
    print('숫자가 큽니다. 더 작은 숫자를 입력하세요.')
  elif user < rd :
    cnt +=1
    print('숫자가 작습니다. 더 큰 숫자를 입력하세요.')
  elif user == rd :
    cnt +=1
    print('정답은 '+str(rd)+'입니다. '+str(cnt)+'번에 맞히셨습니다.')
    break
  else :
    pass

