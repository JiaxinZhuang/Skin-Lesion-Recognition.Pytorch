import csv
import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import random
import scipy.io

def make_dataset(itrno, data_dir):
    images = []
    lblarr_train = []
    lblarr_test = []
    train_data = []
    test_data = []

    train_csv = 'split_data/itr_' + str(itrno) + '_train.csv'
    fn = os.path.join(data_dir, train_csv)
    print('Loading train from {}'.format(fn))
    file = open(fn, 'r')
    for line in file:
        line = line.split(',')
        item = (line[0], int(line[1]))
        lblarr_train.append(int(line[1]))
        train_data.append(item)
    file.close()

    test_csv = 'split_data/itr_' + str(itrno) + '_test.csv'
    fn = os.path.join(data_dir, test_csv)
    print('Loading test from {}'.format(fn))
    file = open(fn, 'r')
    for line in file:
        line = line.split(',')
        item = (line[0], int(line[1]))
        lblarr_test.append(int(line[1]))
        test_data.append(item)
    file.close()

    unique_labels = np.unique(lblarr_train).tolist()
    #count = []
    #for lbl in unique_labels:
    #    ntrain = sum(1 for x in lblarr_train if x==lbl)
    #    ntest = sum(1 for x in lblarr_test if x == lbl)
    #    tot = ntrain + ntest
    #    count.append(tot)
    #    print(f"Number of images in class  {lbl} is  {tot} : {ntrain}, {ntest}")
    #print(f'Total train: {len(lblarr_train)}, test: {len(lblarr_test)}')

    #sv = sum(count)
    #weights = []
    #tot = 0
    #for i, val in enumerate(count):
    #    weights.append((sv-val)/val)
    #    tot = tot + weights[i]
    #for val in weights:
    #   print(np.round(val/tot,3), end=',')
    #print()

    #for val in count:
    #   print(val, end=',')
    #print()

    return train_data, test_data, unique_labels


class DatasetFolder(data.Dataset):
    def __init__(self, train=True, transform=None, transform_target=None, iterNo=1, data_dir='../data/ISIC2018/'):
        self.train_data, self.test_data, self.classes = make_dataset(iterNo, data_dir=data_dir)
        self.transform = transform
        self.train = train # training set or test set

        train_data = 'ISIC2018_Task3_Training_Input'
        #train_data = 'wpr1'
        self.train_data_dir = os.path.join(data_dir, train_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.train:
            path, target = self.train_data[index]
        else:
            path, target = self.test_data[index]

        path = os.path.join(self.train_data_dir, path)
        path += '.jpg'
        imagedata = default_loader(path)
        if self.transform is not None:
            imagedata = self.transform(imagedata)

        #[tmp,path] = os.path.split(path)
        #path = path.split('.')[0]

        return imagedata, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def get_weights_for_balanced_classes(self):
        nclasses = len(self.classes)
        count = [0] * nclasses
        for item in self.train_data:
            count[item[1]] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N/float(count[i])
        weight = [0] * len(self.train_data)
        for idx, val in enumerate(self.train_data):
            weight[idx] = weight_per_class[val[1]]
        return weight

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

#make_dataset(0.5)


class testsetFolder(data.Dataset):
    def __init__(self, transform=None, data_dir='../data/ISIC2018/ISIC2018_Task3_Validation_Input'):
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

