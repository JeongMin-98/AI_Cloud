# 딥러닝/클라우드 기말과제 - 사진 파일 분류하기

<aside>
💡 **기말 과제 목표:** 1,125개의 날씨 사진을 분류하는 CNN 모델을 만들고 그 모델의 성능을 향상시키는 방향을 제시

</aside>

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%AE%E1%84%83%E1%85%B3%20%E1%84%80%E1%85%B5%E1%84%86%E1%85%A1%E1%86%AF%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20-%20%E1%84%89%E1%85%A1%E1%84%8C%E1%85%B5%E1%86%AB%20%E1%84%91%E1%85%A1%E1%84%8B%E1%85%B5%E1%86%AF%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%92%E1%85%A1%20f2c9ba12b25e4046aab2669107a8099e/Untitled.png)

# 👀 CNN 이란?

> CNN 모델은 이미지를 분류하기 용이한 모델이다. 
이미지를 Conv 연산과 Pooling layer로 계층화하여 연산시키고 출력층에 Fully connected하여 Affine한 신경망이다. 
다른 Fully connected layer와는 다른 점으로 Conv연산과 Pooling layer를 사용한다.
> 

---

# 💭 사용 개념

> Convolution 연산
pooling
Image Augmentation
Data Load 성능 개선
Dropout
> 

---

# 🛫 플랜

> MLops 를 사용하여 모델을 설계, 피드백할 예정이다. 
모델의 성능 지표는 Training Error와 Test Error의 차이가 가장 적으며 Test Error가 가장 낮은 모델을 선택할 것이다. 
모델의 성능 개선은 Image Augmentation으로 데이터 표본의 수를 늘리고, Dropout하여 과적합을 방지한다.
> 

# 라이브러리 설정(가상환경 설정)

> python ≥ 3.7
numpy ≥ 1.23
tensorflow ≥ 2 (gpu 버전)
> 

# 데이터 전처리

과제에 주어진 이미지의 개수는 1,125개이다. 해당 모델은 각 label에 따라 이미지를 분류한다. 그렇기 때문에 이미지의 label인 cloudy, rain, shine, sunrise로 하위 디렉토리로 나누어 사진들을 정리한다. 

다음과 같이 디렉터리 구조를 만든다.

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%AE%E1%84%83%E1%85%B3%20%E1%84%80%E1%85%B5%E1%84%86%E1%85%A1%E1%86%AF%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20-%20%E1%84%89%E1%85%A1%E1%84%8C%E1%85%B5%E1%86%AB%20%E1%84%91%E1%85%A1%E1%84%8B%E1%85%B5%E1%86%AF%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%92%E1%85%A1%20f2c9ba12b25e4046aab2669107a8099e/Untitled%201.png)

이미지의 입력 사이즈는 180 * 180로 정하였다. 

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%AE%E1%84%83%E1%85%B3%20%E1%84%80%E1%85%B5%E1%84%86%E1%85%A1%E1%86%AF%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20-%20%E1%84%89%E1%85%A1%E1%84%8C%E1%85%B5%E1%86%AB%20%E1%84%91%E1%85%A1%E1%84%8B%E1%85%B5%E1%86%AF%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%92%E1%85%A1%20f2c9ba12b25e4046aab2669107a8099e/Untitled%202.png)

- 이 API는 디렉토리에 하위 디렉토리에 class_a, class_b가 있다면, 하위디렉토리가 label이 되며, 각각 class_a, class_b에 0, 1로 대응되는 labels을 가지는 tr.data.dataset을 반환한다.
- subset 인자는 해당 데이터가 validation data인지, training data인지 나눈다.
- label_mode 인자는 해당 데이터의 label를 int, catergorical, binary(이진분류)인지 결정한다.
- 이 api를 다음과 같이 이용하여 데이터를 train:test로 나누어 모델에 사용한다.

코드 부분에서 다음과 같이 코드를 짜 train:test(validation)을 7:3 비율로 나눈다. 

```python
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
```

# 모델 구성

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%AE%E1%84%83%E1%85%B3%20%E1%84%80%E1%85%B5%E1%84%86%E1%85%A1%E1%86%AF%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20-%20%E1%84%89%E1%85%A1%E1%84%8C%E1%85%B5%E1%86%AB%20%E1%84%91%E1%85%A1%E1%84%8B%E1%85%B5%E1%86%AF%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%92%E1%85%A1%20f2c9ba12b25e4046aab2669107a8099e/Untitled%203.png)

1. 이미지가 입력되기 전 모든 데이터는 Rescalling 과정을 거친다. 
2. 처음 모델 학습의 epoch 수는 50으로 지정
3. 모델 결과의 정확도와 학습 곡선은 다음과 같다.

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%AE%E1%84%83%E1%85%B3%20%E1%84%80%E1%85%B5%E1%84%86%E1%85%A1%E1%86%AF%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20-%20%E1%84%89%E1%85%A1%E1%84%8C%E1%85%B5%E1%86%AB%20%E1%84%91%E1%85%A1%E1%84%8B%E1%85%B5%E1%86%AF%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%92%E1%85%A1%20f2c9ba12b25e4046aab2669107a8099e/Untitled%204.png)

1. 해당 그래프를 확인한 결과 epoch수가 50번은 너무 많은 것으로 판단되었다.
2. 유의미한 학습 그래프를 가지기 위해서는 early stopping을 하여 epoch수를 20번으로 지정하기로 하였다.
- 해당 모델의 정확도는 매우 높은 수준에 근사하나 이 기본 모델은 학습 모델과 검증 모델의 정확도 차이가 확연하게 벌어지는 것을 확인할 수 있다. 이는 과대적합 징후를 나타낸다. 그렇기 때문에 dropout과 data augumentation을 적용하기로 하였다.

# 모델 최적화

## 데이터 증강

<aside>
💡 데이터 증강 (Data Augmetation)은 이미지를 회전, 반전, 확대, 축소, 색조 변환등 여러가지 이미지 변환을 통해 기존 데이터보다 많은 데이터의 양을 확보하여 모델의 과대적합을 방지하도록 한다

</aside>

데이터 증강을 하기위해서 input layer 단계 이전에 다음과 코드를 삽입한다. 

```python
data_augmentation = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip('horizontal',
                                                     input_shape=(img_height, img_width, 3)),
        layers.experimental.preprocessing.RandomFlip('vertical'),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomZoom(0.2)
    ]
)
```

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%AE%E1%84%83%E1%85%B3%20%E1%84%80%E1%85%B5%E1%84%86%E1%85%A1%E1%86%AF%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20-%20%E1%84%89%E1%85%A1%E1%84%8C%E1%85%B5%E1%86%AB%20%E1%84%91%E1%85%A1%E1%84%8B%E1%85%B5%E1%86%AF%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%92%E1%85%A1%20f2c9ba12b25e4046aab2669107a8099e/Untitled%205.png)

데이터 증강 기법을 사용하게되면 기존에 가지고 있던 이미지보다 더 많은 이미지를 얻을 수 있다. 

## Dropout

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%AE%E1%84%83%E1%85%B3%20%E1%84%80%E1%85%B5%E1%84%86%E1%85%A1%E1%86%AF%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20-%20%E1%84%89%E1%85%A1%E1%84%8C%E1%85%B5%E1%86%AB%20%E1%84%91%E1%85%A1%E1%84%8B%E1%85%B5%E1%86%AF%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%92%E1%85%A1%20f2c9ba12b25e4046aab2669107a8099e/Untitled%206.png)

모델의 계층마다 dropout을 적용시키면 무작위로 노드를 0으로 설정한다. 

```python
data_augmentation = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip('horizontal',
                                                     input_shape=(img_height, img_width, 3)),
        layers.experimental.preprocessing.RandomFlip('vertical'),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomZoom(0.2)
    ]
)

# 위 코드는 모델의 부족한 데이터 표본의 수를 보충해주기 때문에 training loss를 줄일 수 있다. 

```

해당 과정을 거친 이후 모델의 성능 곡선이다. 

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%AE%E1%84%83%E1%85%B3%20%E1%84%80%E1%85%B5%E1%84%86%E1%85%A1%E1%86%AF%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20-%20%E1%84%89%E1%85%A1%E1%84%8C%E1%85%B5%E1%86%AB%20%E1%84%91%E1%85%A1%E1%84%8B%E1%85%B5%E1%86%AF%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%92%E1%85%A1%20f2c9ba12b25e4046aab2669107a8099e/Untitled%207.png)

과적합이 해결된 모습이고 기존 모델에 비해 traiing 모델의 정확도와 Validation 모델 정확도의 차이가 크게 줄어들었다. 그리고 loss 그래프에 training loss와 validation loss의 차이인 generalization gap이 줄어들어 두 모델의 성능이나 정확도 측면에서 모두 개선된 것을 확인할 수 있다. 

# 최종 모델 구성

최종 모델에 대한 코드는 다음과 같다. 

```python
data_augmentation = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip('horizontal',
                                                     input_shape=(img_height, img_width, 3)),
        layers.experimental.preprocessing.RandomFlip('vertical'),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomZoom(0.2)
    ]
)

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

tf.saved_model.save(model, 'C:\\Users\\jeong\\Desktop\\정민\\3학년 2학기\\[DKU]딥러닝_클라우드\\final\\model')
```

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%AE%E1%84%83%E1%85%B3%20%E1%84%80%E1%85%B5%E1%84%86%E1%85%A1%E1%86%AF%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20-%20%E1%84%89%E1%85%A1%E1%84%8C%E1%85%B5%E1%86%AB%20%E1%84%91%E1%85%A1%E1%84%8B%E1%85%B5%E1%86%AF%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%92%E1%85%A1%20f2c9ba12b25e4046aab2669107a8099e/Untitled%208.png)

# 전체 소스코드

```python
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

data_dir = Path('DeepLearning_Cloud/final/dataset2')
data_dir = Path('C:\\Users\\jeong\\PycharmProjects\\ML_algorithm\\DeepLearning_Cloud/final/dataset2')

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

tf.saved_model.save(model, 'C:\\Users\\jeong\\Desktop\\정민\\3학년 2학기\\[DKU]딥러닝_클라우드\\final\\model')
```