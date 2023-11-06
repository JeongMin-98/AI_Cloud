import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split


# Q1 lstat (소득분위가 하위인 사람들의 비율) 로
# medv (주택가격)을 예측하는 단순 선형회귀 모델을 만드시오 (train, test 나누지 않음). 모델의 내용을 보이시오

# lstat => 독립변수, medv 종속변수

BostonHousing = pd.read_csv("DeepLearning_Cloud\BostonHousing.csv")

lstat = BostonHousing["lstat"]
medv = BostonHousing["medv"]

lstat = np.array(lstat).reshape(-1, 1)
medv = np.array(medv).reshape(-1, 1)

model = LinearRegression()
model.fit(lstat, medv)

print('Coefficients: {0:.2f}, Intercept {1:.3f}'.format(model.coef_[0][0], model.intercept_[0]))
# Q2. 모델에서 만들어진 회귀식을 쓰시오 (medv = W x lstat + b 의 형태)
print('y={0:.2f}x+{1:.3f}'.format(model.coef_[0][0], model.intercept_[0]))

# Q3. 회귀식을 이용하여 lstat 의 값이 각각 2.0, 3.0, 4.0, 5.0 일 때 medv 의 값을 예측하여 제시하시오.
lstat_2 = model.predict([[2.0]])
lstat_3 = model.predict(([[3.0]]))
lstat_4 = model.predict(([[4.0]]))
lstat_5 = model.predict([[5.0]])
print("lstat:{0} medv:{1:.3f}".format(2.0, lstat_2[0][0]))
print("lstat:{0} medv:{1:.3f}".format(3.0, lstat_3[0][0]))
print("lstat:{0} medv:{1:.3f}".format(4.0, lstat_4[0][0]))
print("lstat:{0} medv:{1:.3f}".format(5.0, lstat_5[0][0]))

# Q4. 데이터셋의 모든 lstat 값을 회귀식에 넣어 mdev 의 값을 예측 한 뒤 mean square error를 계산하여 제시하시오
pred_lstat = model.predict(lstat)
print('Mean Squared error: {0:.2f}'.format(mean_squared_error(pred_lstat, medv)))

# Q5. lstat (소득분위가 하위인 사람들의 비율), ptratio(초등교사비율), tax(세금), rad(고속도로접근성)로
# medv (주택가격)을 예측하는 단순 선형회귀 모델을 만드시오 (tain, test 나누지 않음).
# 모델의 내용을 보이시오

df_X = BostonHousing[['lstat', 'ptratio', 'tax', 'rad']]
df_y = BostonHousing['medv']

Multiple_model = LinearRegression()
Multiple_model.fit(df_X, df_y)

print('Coefficients: {0:.2f}, {1:.2f}, {2:.2f} Intercept {3:.3f}'.format(Multiple_model.coef_[0], Multiple_model.coef_[1], Multiple_model.coef_[2], Multiple_model.intercept_))

# Q7. lstat, ptratio, tax, rad 의 값이 다음과 같을 때 mdev 의 예측값을 보이시오.

data_example = pd.DataFrame({'lstat': [2.0,3.0,4.0],
                             'ptratio': [14,15,16],
                             'tax':[296,222,250],
                             'rad':[1,2,3]})


print(Multiple_model.predict(data_example))
pred_y = Multiple_model.predict(df_X)
print('Mean Squared error: {0:.2f}'.format(mean_squared_error(pred_y, df_y)))

# Q10. gre,  gpa, rank를 가지고 합격여부를 예측하는 logistic regression 모델을 만드시오.
# (train, test를 나누되 test 의 비율은 30% 로 하고 random_state 는 1234 로 한다)

ucla_admit = pd.read_csv('DeepLearning_Cloud\_ucla_admit.csv')

df_ucla_X = ucla_admit[['gre','gpa','rank']]
df_ucla_y = ucla_admit['admit']

train_X, test_X, train_y, test_y = train_test_split(df_ucla_X, df_ucla_y, test_size=0.3, random_state=1234)
Logistic_model = LogisticRegression()
Logistic_model.fit(train_X, train_y)

pred_y = Logistic_model.predict(test_X)
acc = accuracy_score(pred_y, test_y)
print('정확도 : {:.3f}'.format(acc))

# Q11. 모델을 테스트 하여 training accuracy 와 test accuracy를 보이시오
pred_test_y = Logistic_model.predict(test_X)
pred_train_y = Logistic_model.predict(train_X)
train_acc = accuracy_score(pred_train_y, train_y)
test_acc = accuracy_score(pred_test_y, test_y)
print('train_set 정확도 : {0:.3f} \n test_set 정확도 : {1:.3f}'.format(train_acc, test_acc))



#Q12. gre,  gpa, rank 가 다음과 같을 때 합격 여부를 예측하여 보이시오
admit_example = pd.DataFrame({'gre': [400, 550, 700],
                             'gpa': [3.5, 3.8, 4.0],
                             'rank':[5, 2, 2]})

print(Logistic_model.predict(admit_example))

# Q13.이번에는 gre,  gpa만 가지고 합격 여부를 예측하는 모델을 만드시오
# (train, test를 나누되 test 의 비율은 30% 로 하고 random_state 는 1234 로 한다)

ucla_admit = pd.read_csv('DeepLearning_Cloud\_ucla_admit.csv')

df_ucla_2_X = ucla_admit[['gre','gpa']]
df_ucla_2_y = ucla_admit['admit']
#앞서 나온 데이터 셋과 구분하기 위해서 2를 넣음
train_2_X, test_2_X, train_2_y, test_2_y = train_test_split(df_ucla_2_X, df_ucla_2_y, test_size=0.3, random_state=1234)
Logistic_model = LogisticRegression()
Logistic_model.fit(train_2_X, train_2_y)

pred_2_y = Logistic_model.predict(test_2_X)
acc_2 = accuracy_score(pred_2_y, test_2_y)
print('정확도 : {:.3f}'.format(acc_2))

# Q14. 모델을 테스트 하여 training accuracy 와 test accuracy를 보이시오
pred_test_2_y = Logistic_model.predict(test_2_X)
pred_train_2_y = Logistic_model.predict(train_2_X)
train_acc_2 = accuracy_score(pred_train_2_y, train_2_y)
test_acc_2 = accuracy_score(pred_test_2_y, test_2_y)
print('train_set 정확도 : {0:.3f} \n test_set 정확도 : {1:.3f}'.format(train_acc_2, test_acc_2))


