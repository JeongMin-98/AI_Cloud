import numpy as np
from PIL import Image


def tensor2image(tensor):
    data = np.array(tensor)
    # (C, H, W) -> (H, W, C)
    img = np.transpose(data, (1, 2, 0)) * 255
    img = np.array(img, dtype=np.uint8)

    img = Image.fromarray(img)
    img.show()
