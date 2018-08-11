import csv
import os
import numpy as np
import torch.utils.data as data
import random
import torch
from PIL import Image

class testsetFolder(data.Dataset):
    def __init__(self, transform=None, data_dir='./ISIC2018_Task3_Validation_Input'):
        self.transform = transform
        filenames = os.listdir(data_dir)
        filenames = list(filter(lambda x: x.split('.') [-1] == 'jpg', filenames))
        self.data = [os.path.join(data_dir, x) for x in filenames]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.data[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        filename = os.path.split(path)[-1]
        filename = filename.split('.')[0]
        return filename, sample

    def __len__(self):
        return len(self.data)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)