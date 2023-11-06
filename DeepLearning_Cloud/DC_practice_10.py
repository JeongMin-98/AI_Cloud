from keras import optimizers, Input
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(34)


"""
Q1 (7점) 제공된 PimaIndiansDiabetes.csv 파일에 대해 Keras를 이용한 classification 모델을 개발하고 테스트 하시오
train/test set을 나누되 test set 은 전체 dataset 의 30% 로 한다.
hidden layer 의 수는 4개, layer별 노드 수는 각자 정한다.
hidden layer 의 활성화 함수는 relu, output layer 의 노드수는 softmax 로 한다
각 hidden layer 에 dropout을 적용한다. (적용률을 각자 알아서)
기타 필요한 매개변수들은 각자 정한다.
epoch 는 20,40,60,80 으로 변화시켜 가면서 테스트한다.
각 training accuracy 와 validation accuracy 학습곡선 그래프를 제시한다
"""

# df = pd.read_csv("DeepLearning_Cloud\PimaIndiansDiabetes.csv", encoding='cp949')
df = pd.read_csv('PimaIndiansDiabetes.csv')
dataset = df.values
X = dataset[:, :-1].astype(float)
Y = dataset[:, -1]


encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.3, random_state=1234, stratify=dummy_y)

adam = optimizers.Adam(learning_rate=0.01)
model = Sequential()

epoch = [20, 40, 60, 80]

for _epoch in epoch:
    model.add(Dense(32, input_dim=8, activation='relu', kernel_initializer=keras.initializers.glorot_uniform(seed=1234)))
    model.add(Dropout(rate=0.5))  # dropout
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(rate=0.4))  # dropout
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(rate=0.4))  # dropout
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # output layer
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    disp = model.fit(X_train, y_train, batch_size=5, epochs=_epoch, verbose=1, validation_split=0.2)
    pred = model.predict(X_test)
    print(pred)
    y_classes = [np.argmax(y, axis=None, out=None) for y in pred]
    print(y_classes)

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss : ", score[0])
    print("Test accuracy: ", score[1])

    plt.plot()
    plt.plot(disp.history['accuracy'])
    plt.plot(disp.history['val_accuracy'])
    plt.title("model accuracy epoch: {}".format(_epoch))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot()
    plt.plot(disp.history['loss'])
    plt.plot(disp.history['val_loss'])
    plt.title("model accuracy epoch: {}".format(_epoch))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
