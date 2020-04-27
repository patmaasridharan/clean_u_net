import numpy as np
import torch.utils.data as data
from PIL import Image

import torch
import numpy as np

def image_to_tensor(image, mean=0, std=1.):

    image = image.astype(np.float32)
    image = (image - mean) / std
    image = np.resize(image, (image.shape[0], image.shape[1], 1))
    image = image.transpose((2, 0, 1))
    tensor = torch.from_numpy(image)
    return tensor


class TestImageDataset(data.Dataset):
    def __init__(self, X_data, img_resize=(128, 128)):
        self.img_resize = img_resize
        self.X_train = X_data

    def __getitem__(self, index):

        img_path = self.X_train[index]
        img = Image.open(img_path)
        img = img.resize(self.img_resize, Image.ANTIALIAS)
        img = np.asarray(img.convert("L"), dtype=np.float32) # 

        img = image_to_tensor(img)
        return img, img_path.split("/")[-1]

    def __len__(self):
        return len(self.X_train)
