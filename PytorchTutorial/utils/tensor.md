# Tensors

날짜: 2023년 10월 26일
태그: Pytorch

# Tensors

PyTorch의 Tensors는 다차원 배열로, 수치 연산을 위한 핵심 데이터 구조입니다. Tensors는 다차원 텐서의 크기, 형상, 데이터 유형 등을 표현합니다. PyTorch에서 Tensors는 GPU 상에서도 연산이 가능하며, 자동 미분을 지원하여 딥러닝 모델의 학습에 유용하게 사용됩니다.

Tensors는 다양한 수학적 연산을 제공하며, 행렬 연산, 선형 대수, 통계 분석 등 다양한 분야에서 활용됩니다. 또한, Tensors는 신경망 모델의 입력 데이터, 가중치, 편향 등을 표현하고 처리하는 데에도 사용됩니다.

PyTorch에서 Tensors를 사용하기 위해서는 먼저 PyTorch 라이브러리를 설치하고, `torch` 모듈을 import해야 합니다. Tensors는 `torch.Tensor` 클래스를 통해 생성되며, 데이터는 일반적으로 NumPy 배열이나 Python 리스트로부터 생성될 수 있습니다.

PyTorch의 Tensors는 다양한 연산을 지원하며, 텐서간의 산술 연산, 선형 대수 연산, 행렬 연산 등을 수행할 수 있습니다. 또한, GPU를 사용하여 Tensors 연산을 가속화할 수도 있습니다.

## Tensor 초기화(생성)

### data로부터 직접적으로 생성(list)

예를 들어, 다음과 같이 Tensors를 생성할 수 있습니다:

```python
import torch

# Python 리스트로부터 utils 생성
data = [1, 2, 3, 4, 5]
tensor = torch.Tensor(data)
```

### NumPy array로부터 생성

```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

### 다른 Tensor로부터 생성

```python
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones utils: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random utils: \n {x_rand} \n")
```

### 랜덤 값 또는 상수 값 Tensor 생성하기

```python
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random utils: \n {rand_tensor} \n")
print(f"Ones utils: \n {ones_tensor} \n")
print(f"Zeros utils: \n {zeros_tensor}")
```

## Tensor의 속성

Tensor의 속성값에는 shape, datatpye 그리고 GPU에 저장되어있는지 아닌지 나타내는 값이 있다. 

 

```python
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

## Slicing and Indexing

```python
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
```

## Tensor의 연산

PyTorch의 Tensors는 다양한 연산을 지원합니다. 다음은 Tensor의 주요 연산들입니다:

### 산술 연산

Tensor 간의 산술 연산은 일반적인 수학 연산과 동일한 방식으로 수행됩니다. 예를 들어, 덧셈, 뺄셈, 곱셈, 나눗셈 연산을 수행할 수 있습니다.

```python
# utils 덧셈 연산
result = tensor1 + tensor2

# utils 뺄셈 연산
result = tensor1 - tensor2

# utils 곱셈 연산
result = tensor1 * tensor2

# utils 나눗셈 연산
result = tensor1 / tensor2

```

### 선형 대수 연산

PyTorch의 Tensors는 선형 대수 연산도 지원합니다. 예를 들어, 행렬 곱셈, 전치행렬, 역행렬 등의 연산을 수행할 수 있습니다.

```python
# 행렬 곱셈 연산
result = torch.matmul(matrix1, matrix2)
result = matrix1 @ matrix2

# 전치행렬 연산
result = tensor1.T

# 역행렬 연산
result = torch.inverse(matrix)

```

### 행렬 연산

PyTorch의 Tensors는 다양한 행렬 연산을 수행할 수 있습니다. 예를 들어, 행렬 합, 행렬 분리, 행렬 슬라이스 등이 있습니다.

```python
# 행렬 합 연산
result = torch.sum(matrix)

# 행렬 분리 연산
result1, result2 = torch.split(matrix, split_size, dim)

# 행렬 슬라이스 연산
result = matrix[start:end, start:end]

```

### Joining Tensors

torch.cat을 사용하여 정한 dimension으로 tensor를 합칠 수 있다. 

```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```

### 하나의 요소를 가진 Tensor → Value

하나의 요소만 가진 tensor를 합산하여 Value로 바꿀 수 있다. 

```python
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
```

이 외에도 Tensors는 다양한 수학적 연산을 지원하며, 통계 분석 등 다양한 분야에서 활용할 수 있습니다.