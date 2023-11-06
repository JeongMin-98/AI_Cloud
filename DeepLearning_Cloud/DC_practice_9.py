from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("liver.csv")
dataset = df.values

X = dataset[:, 1:].astype(float)
Y = dataset[:, 0]


encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

dummy_y = np_utils.to_categorical(encoded_Y)

X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.4, random_state=1234)

model = Sequential()
model.add(Dense(10, input_dim=6, activation='relu'))
model.add(Dense(10, activation='relu'))
# Q 2 hidden layer add
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

tf.random.set_seed(2)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


epochs = [50, 100, 150, 200]

for _epochs in epochs:
    tf.random.set_seed(2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    disp = model.fit(X_train, y_train, batch_size=10, epochs=_epochs, verbose=1, validation_data=(X_test, y_test))

    pred = model.predict(X_test)
    # print(pred)
    # y_classes = [np.argmax(y, axis=None, out=None) for y in pred]
    # print(y_classes)

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss: ', score[0])
    print('test accuracy: ', score[1])


    plt.plot(disp.history['val_loss'])
    plt.plot(disp.history['val_accuracy'])
    plt.title('model loss and accuary : {}'.format(_epochs))
    plt.ylabel('accuracy, loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'accuracy'], loc='upper left')
    plt.show()