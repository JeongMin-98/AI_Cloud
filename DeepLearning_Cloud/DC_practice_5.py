from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd

df = pd.read_csv("DeepLearning_Cloud\PimaIndiansDiabetes.csv")

print(df.columns)



df_X = df.loc[:, df.columns != 'diabetes']
df_y = df['diabetes']

train_X, test_X, train_y, test_y = train_test_split(df_X, df_y, test_size=0.3, random_state=1234)

DT_model = DecisionTreeClassifier(random_state=1234)

DT_model.fit(train_X, train_y)

print('Train accuracy : ', DT_model.score(train_X, train_y))
print('Test accuracy : ', DT_model.score(test_X, test_y))
accuracy = []
DT_model_accuracy = cross_val_score(DT_model, df_X, df_y, cv = 10)
accuracy.append(DT_model_accuracy.mean())

RF_model = RandomForestClassifier(random_state=1234)
RF_model_accuarcy = cross_val_score(RF_model, df_X, df_y, cv=10)
accuracy.append(RF_model_accuarcy.mean())

svm = svm.SVC() #rbf default parameter
svm_accuarcy = cross_val_score(svm, df_X, df_y, cv=10)
accuracy.append(svm_accuarcy.mean())
print("각 모델들의 10-fold cross validation결과\n")
accuracy_list = ["Decision Tree", "Random Forest", "support vector machine"]
for i in range(len(accuracy)):
    print("{0}의 평균 accuracy 값 : {1:.2f}".format(accuracy_list[i], accuracy[i]))


#Q2

from sklearn import svm
from sklearn.model_selection import cross_val_score
import pandas as pd

df = pd.read_csv("DeepLearning_Cloud\PimaIndiansDiabetes.csv")

df_X = df.loc[:, df.columns != 'diabetes']
df_y = df['diabetes']

kernel = ['poly','linear','rbf', 'sigmoid']
_model = []

kernel_accuracy_mean = []
for _kernel in kernel:
    _model.append(svm.SVC(kernel=_kernel))

for model in _model:
    kernel_accuracy= cross_val_score(model, df_X, df_y, cv=10)
    kernel_accuracy_mean.append(kernel_accuracy.mean())

for i in range(len(kernel)):
    print("{0}의 평균 accuracy 값 : {1:.2f}".format(kernel[i], kernel_accuracy_mean[i]))


#Q3

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd


df = pd.read_csv("DeepLearning_Cloud\PimaIndiansDiabetes.csv")

df_X = df.loc[:, df.columns != 'diabetes']
df_y = df['diabetes']

n_estimators = [100,200,300,400,500]
max_features = [1,2,3,4,5]

model_accuracy = {}
accuracy_list = []

for n_est in n_estimators:
    for feat in max_features:

        model = RandomForestClassifier(n_estimators=n_est, max_features=feat)

        accuracy = cross_val_score(model, df_X, df_y)

        mean_accuracy = accuracy.mean()

        model_accuracy[str(n_est)+" " + str(feat)] = mean_accuracy


for key, value in model_accuracy.items():
    print("RF classifier의 파라미터는 {0}의 10겹 교차 검증의 평균 정확도는 {1:.3f}".format(key, value))

print(" 가장 mean_accuracy가 높은 모델은 {0}를 사용한 모델이다.".format(max(model_accuracy, key=model_accuracy.get)))



