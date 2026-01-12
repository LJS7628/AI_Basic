class sadan :

  def __init__(self,modelNO, seName, price, efficiency, brand, segment, curSpeed) :
    self.modelNO = modelNO
    self.seName = seName
    self.price = price
    self.efficiency = efficiency
    self.brand = brand
    self.segment = segment
    self.curSpeed = curSpeed

  def showCarInfo(self) :
    print('modelNO : ' + str(self.modelNo))
    print('seName : '+str(self.seName))
    print('price : '+str(self.price))
    print('efficiency : '+str(self.efficiency))
    print('brand : '+str(self.brand))
    print('segment : '+str(self.segment))
    print('curSpeed : '+str(self.curSpeed))

  def getDistance(self,gas) :
    result=round(self.efficiency*gas)
    print(str(self.seName)+" : "+str(result)+"킬로미터를 갈 수 있어요.")

  def speedUp(self) :
    return self.curSpeed+5

  def speedDown(self) :
    return self.curSpeed-5

  def setPrice(self,price) :
    self.price = price

  def getPrice(self) :
    return self.price

  def getSeName(self) :
    return self.seName

data1 = sadan('H_001', '모닝', 9000000, 15.6, '기아', '소형', 0)
data2 = sadan('S_002', 'SM-5', 25000000, 12.1, '르노삼성', '중형', 0)
data3 = sadan('M_001', 'E220d', 80000000, 11.6, '벤츠', '중형', 0)
data4 = sadan('H_002', '소나타', 27000000, 13.1, '현대', '중형', 0)

cars = list()
cars.append(data1)
cars.append(data2)
cars.append(data3)
cars.append(data4)

total = 0
mean = 0
for i in cars :
  i.getDistance(13)
  total = total+i.getPrice()
  mean = total/len(cars)


print("각 차량의 가격 총합 : "+str(total))
print("차량의 가격 평균 : "+str(mean))

car_price = list()
for i in cars :
  car_price.append(i.getPrice())

for i in cars :
  if(i.getPrice() == min(car_price)) :
    print("최저가 차량 : "+str(i.getSeName())+" : "+str(i.getPrice()))
  if(i.getPrice() == max(car_price)) :
    print("최고가 차량 : "+str(i.getSeName())+" : "+str(i.getPrice()))

