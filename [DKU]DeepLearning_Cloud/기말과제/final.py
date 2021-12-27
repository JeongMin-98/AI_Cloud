
"""
    딥러닝/클라우드 기말 과제
    32170939 김정민
    jeongmin981@gmail.com

    참고자료
    tensorflow.org

"""


import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import PIL

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from pathlib import Path

# tf random seed
tf.random.set_seed(123)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] ='3'

# dataset 폴더를 불러오기 전 사진 이름에 따라 폴더별로 정리.
"""
    dataset/
        cloudy/
        rain/
        shine/
        sunrise/
        
"""

data_dir = Path('[DKU]DeepLearning_Cloud/기말과제/dataset2')
data_dir = Path('C:\\Users\\jeong\\PycharmProjects\\ML_algorithm\\[DKU]DeepLearning_Cloud/기말과제/dataset2')

image_list = list(data_dir.glob('*/*'))
print('이미지의 개수 : {}'.format(len(image_list)))


# batch_size 처음은 32로 정함
batch_size = 32
img_height = 180
img_width = 180

# Data_dir에 있는 이미지 파일들을 tf.data.dataset으로 형성
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_height,img_width),
    seed=123,
    validation_split=0.3,
    subset='training',
)


val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_height,img_width),
    seed=123,
    validation_split=0.3,
    subset='validation',
)

class_names = train_ds.class_names
print(class_names)


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        idx = 0
        for j in range(len(labels[i])):
            idx += j * labels[i][j]
        idx = idx.numpy().astype('uint8')
        plt.title(class_names[idx])
        plt.axis('off')
plt.show()

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = 4

# 초기 모델
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
""" 
    from_logits 파라미터는 보통 True으로 설정하여 모델을 compile한다. 만약 output_layers가 활성함수 softmax를 거치지 않는
    경우 설정한다. 
    하지만 이 모델에 경우 softmax 활성화 함수를 사용하기 때문에 False로 지정해준다. 
"""
model.summary()

epochs=20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=1
)


# model 성능 평가
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


## validation data에서 loss값이 더이상 줄지않는 과대적합현상이발생
## 이를 해결하기위해 데이터증강(Data Augumentation) 사용

data_augmentation = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip('horizontal',
                                                     input_shape=(img_height, img_width, 3)),
        layers.experimental.preprocessing.RandomFlip('vertical'),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomZoom(0.2)
    ]
)

plt.figure(figsize=(10,10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(augmented_images[0].numpy().astype('uint8'))
        plt.axis("off")
plt.show()

model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1. / 255),
    layers.Conv2D(16, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

epochs = 20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

tf.saved_model.save(model, 'C:\\Users\\jeong\\Desktop\\정민\\3학년 2학기\\[DKU]딥러닝_클라우드\\기말과제\\model')
