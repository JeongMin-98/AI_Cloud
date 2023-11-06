from skimage import io
from tkinter.filedialog import askopenfilename
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Q2 (3점) 컬러 이미지 파일을 하나 지정하여 다음의 작업을 수행하고 결과를 제시하시오
# (1) 컬러 이미지 파일을 화면에 출력한다
# fname = askopenfilename()
image = io.imread('bird.jpeg')

plt.imshow(image)
plt.show()
# (2) 컬러 이미지를 흑백 이미지로 변환하여 화면에화면에출력한다
gray_image = color.rgb2gray(image)
plt.imshow(gray_image)
io.imsave('rgb2gray.jpg',gray_image)
plt.show()
# (3) 컬러 이미지의 크기를 1/3 로 축소하여 출혁한다
from skimage import transform

new_shape = (image.shape[0]//3, image.shape[1]//3, image.shape[2])
small = transform.resize(image=image, output_shape=new_shape)

plt.imshow(small)
plt.show()
io.imsave('small.jpg', small)
# (4) 컬러 이미지를 좌우 반전시켜 화면에 출력한다
flip_img = np.fliplr(image)
plt.imshow(np.fliplr(image))
io.imsave('flip.jpg', flip_img)
plt.show()
# (5) 컬러 이미지를 상하 반전시켜 화면에 출력한다 (ppt에 없는 내용임)
ud_img = np.flipud(image)
plt.imshow(np.flipud(image))
io.imsave('ud.jpg', ud_img)
plt.show()
