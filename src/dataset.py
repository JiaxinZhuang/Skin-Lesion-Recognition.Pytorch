"""Dataset.

    Customize your dataset here.
"""

import os

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from torchvision import transforms


class Skin7(Dataset):
    """SKin Lesion"""
    def __init__(self, root="./data", iter_fold=1, train=True, transform=None):
        self.root = os.path.join(root, "ISIC2018")
        self.transform = transform
        self.train = train

        self.data, self.targets = self.get_data(iter_fold, self.root)
        self.classes_name = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        self.classes = list(range(len(self.classes_name)))
        self.target_img_dict = {}
        targets = np.array(self.targets)
        for target in self.classes:
            indexes = np.nonzero(targets == target)[0]
            self.target_img_dict.update({target: indexes})

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the
                   target class.
        """
        path = self.data[index]
        target = self.targets[index]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def get_data(self, iterNo, data_dir):

        if self.train:
            csv = 'split_data/split_data_{}_fold_train.csv'.format(iterNo)
        else:
            csv = 'split_data/split_data_{}_fold_test.csv'.format(iterNo)

        fn = os.path.join(data_dir, csv)
        csvfile = pd.read_csv(fn)
        raw_data = csvfile.values

        data = []
        targets = []
        for path, label in raw_data:
            data.append(os.path.join(self.root,
                                     "ISIC2018_Task3_Training_Input", path))
            targets.append(label)

        return data, targets


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def pil_loader(path):
    """Image Loader
    """
    with open(path, "rb") as afile:
        img = Image.open(afile)
        return img.convert("RGB")


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def print_dataset(dataset, print_time):
    print(len(dataset))
    from collections import Counter
    counter = Counter()
    labels = []
    for index, (img, label) in enumerate(dataset):
        if index % print_time == 0:
            print(img.size(), label)
        labels.append(label)
    counter.update(labels)
    print(counter)


if __name__ == "__main__":
    root = "../data"
    dataset = Skin7(root=root, train=True, transform=transforms.ToTensor())
    print_dataset(dataset, print_time=1000)

    dataset = Skin7(root=root, train=False, transform=transforms.ToTensor())
    print_dataset(dataset, print_time=1000)
