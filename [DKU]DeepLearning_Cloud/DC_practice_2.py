# Q1

ans = True
while ans:
    question = int(input("화면에 출력하고 싶은 구구단의 단을 입력하시요:  "))

    if question == "":
        break
    elif question:
        for i in range(1, 10):
            answer = question * i
            print("{} * {} = {}".format(question, i , answer))
        break


# Q2
print("화면에 키(cm)와 몸무게(kg)을 입력하시요")
height, weight = map(float,input().split())
height = height / 100
BMI = weight / (height ** 2)

if BMI <= 18.5:
    print("BMI 값:{0:.2f}, 비만 정도: {1}".format(BMI, "저체중"))
elif (BMI > 18.5) & (BMI <= 23):
    print("BMI 값:{0:.2f}, 비만 정도: {1}".format(BMI, "정상"))
elif (BMI > 23) & (BMI <= 25):
    print("BMI 값:{0:.2f}, 비만 정도: {1}".format(BMI, "과체중"))
elif (BMI > 25) & (BMI <= 30):
    print("BMI 값:{0:.2f}, 비만 정도: {1}".format(BMI, "비만"))
else:
    print("BMI 값:{0:.2f}, 비만 정도: {1}".format(BMI, "고도비만"))

# Q 3

step = int(input("높이를 입력하시오: "))

for j in range(1, step+1):
    print("*"* j)


# Q 4
step = int(input("높이를 입력하세요: "))
def downtree(n):

    for j in range(n-1, -1, -1):
        print(" " * j + "*" * (n-j))
    return

downtree(step)
# Q 5

before_arr = [7, 1, 10, 4, 6, 9, 2, 8, 15, 12, 17, 19, 18]
after_arr = []

sig = True
length = 0
while sig:
    min_index = before_arr.index(min(before_arr))
    after_arr.append(before_arr[min_index])
    before_arr[min_index] = 999
    length = length+1
    if len(before_arr) == length:
        print(after_arr)
        break


# Q6

A = [1,2,3]
B = [4,5,6]


def mul_arr(a, b):
    _mul_arr = []
    if len(a) != len(b):
        print("둘의 길이가 같지 않습니다.")
        return
    else:
        for i in range(0, len(a)):
            _mul_arr.append(a[i]*b[i])

    return _mul_arr


# Q 7

import numpy as np
my_arr = np.arange(1, 51, 1)
print(my_arr)
my_arr.shape = (5, 10)
print(my_arr)
my_arr.reshape(2, 25)
print(my_arr)

# Q 8
# (1) my_arr 의 값들에 각각 2를 곱한 결과를 보이시오
print(my_arr*2)
# (2) my_arr 의 값들중 20 이하의 값들에 대해서만 100을 더한 후에 my_arr 에 저장
#     하시오. my_arr 의 내용을 보이시오

my_arr[my_arr <= 20] = my_arr[my_arr <= 20] + 100
print(my_arr)

# (3) my_arr 에서 2,3열의 데이터만 잘라서 보이시오.
my_arr[1:3,:]
print(my_arr[1:3,:])

# (4) my_arr 에서 5~8행의 데이터만 잘라서 보이시오.
my_arr[:,4:8]
print(my_arr[:,4:8]


# Q 9

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
x = np.arange(1,13)
y = np.array([20,22,37,79,90,109,288,277,140,50,48,19])

plt.plot(x,y)
plt.title("월별 강수량")
plt.xlabel("월")
plt.ylabel("강수량")
plt.show()