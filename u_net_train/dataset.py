import numpy as np
import torch.utils.data as data
from PIL import Image

import torch
import numpy as np

def image_to_tensor(image):

    image = image.astype(np.float32)
    image = np.resize(image, (image.shape[0], image.shape[1], 1))
    image = image.transpose((2, 0, 1))
    tensor = torch.from_numpy(image)
    return tensor


def mask_to_tensor(mask, threshold):

    mask = (mask > threshold).astype(np.float32)
    tensor = torch.from_numpy(mask).type(torch.FloatTensor)
    return tensor

class TrainImageDataset(data.Dataset):
    def __init__(self, X_data, y_data=None,
                X_transform=None, threshold=0.5):

        self.threshold = threshold
        self.X_train = X_data
        self.y_train_masks = y_data
        self.X_transform = X_transform

    def __getitem__(self, index):

        img = Image.open(self.X_train[index])
        img = np.asarray(img.convert("L"), dtype=np.float32) # 

        mask = Image.open(self.y_train_masks[index])
        mask = np.asarray(mask.convert("L"), dtype=np.float32)  # 

        if self.X_transform:
            img, mask = self.X_transform(img, mask)

        img = image_to_tensor(img)
        mask = mask_to_tensor(mask, self.threshold)
        
        return img, mask

    def __len__(self):
        assert len(self.X_train) == len(self.y_train_masks)
        return len(self.X_train)
