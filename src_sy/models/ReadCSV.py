import numpy as np
import csv
import os
import torch.utils.data as data
import random
import torch
from PIL import Image

def get_five_fold(datadir):
    first_tuple = []
    second_tuple = []
    third_tuple = []
    four_tuple = []
    five_tuple = []
    with open(os.path.join('../data/ISIC2018/', 'split_data.csv'), 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for line_no, row in enumerate(reader):
            if line_no == 0: continue
            row = row[0].split(',')
            if row[0] != '':
                first_fold_name = os.path.join(datadir, row[0]+'.jpg')
                first_fold_label = int(float(row[1]))
                first_item = (first_fold_name, first_fold_label)
                first_tuple.append(first_item)

            if row[2] != '':
                second_fold_name = os.path.join(datadir, row[2] + '.jpg')
                second_fold_label = int(float(row[3]))
                second_item = (second_fold_name, second_fold_label)
                second_tuple.append(second_item)

            if row[4] != '':
                third_fold_name = os.path.join(datadir, row[4] + '.jpg')
                third_fold_label = int(float(row[5]))
                third_item = (third_fold_name, third_fold_label)
                third_tuple.append(third_item)

            if row[6] != '':
                four_fold_name = os.path.join(datadir, row[6] + '.jpg')
                four_fold_label = int(float(row[7]))
                four_item = (four_fold_name, four_fold_label)
                four_tuple.append(four_item)

            if row[8] != '':
                five_fold_name = os.path.join(datadir, row[8] + '.jpg')
                five_fold_label = int(float(row[9]))
                five_item = (five_fold_name, five_fold_label)
                five_tuple.append(five_item)
    first_train_data = first_tuple+second_tuple+third_tuple+four_tuple
    first_test_data = five_tuple
    second_train_data = first_tuple+second_tuple+third_tuple+five_tuple
    second_test_data = four_tuple
    third_train_data = first_tuple+second_tuple+four_tuple+five_tuple
    third_test_data = third_tuple
    four_train_data = first_tuple+third_tuple+four_tuple+five_tuple
    four_test_data = second_tuple
    five_train_data = second_tuple+third_tuple+four_tuple+five_tuple
    five_test_data = first_tuple

    return first_train_data, first_test_data, second_train_data, second_test_data, third_train_data, third_test_data, four_train_data, four_test_data, five_train_data, five_test_data


#first_train_data, first_test_data, second_train_data, second_test_data, third_train_data, third_test_data, four_train_data, four_test_data, five_train_data, five_test_data = get_five_fold(datadir)
#a = set(first_train_data).intersection(second_train_data)
#a = set(first_train_data).intersection(third_train_data)
#a = set(first_train_data).intersection(four_train_data)
#a = set(first_train_data).intersection(five_train_data)
#a = set(first_train_data).intersection(first_train_data)
#print(len(a))
#b = a


class DatasetFolder(data.Dataset):
    def __init__(self, train=True, transform=None, iterNo=1, data_dir='./data/ISIC2-18/ISIC2018_Task3_Training_Input/'):
        """
            fold_index: 1-5
        """
        first_train_data, first_test_data, second_train_data, second_test_data, third_train_data, third_test_data, four_train_data, four_test_data, five_train_data, five_test_data = get_five_fold(
            data_dir)
        self.transform = transform
        self.fold_index = iterNo
        if self.fold_index == 1:
            self.train_data = first_train_data
            self.test_data = first_test_data
        elif self.fold_index == 2:
            self.train_data = second_train_data
            self.test_data = second_test_data
        elif self.fold_index == 3:
            self.train_data = third_train_data
            self.test_data = third_test_data
        elif self.fold_index == 4:
            self.train_data = four_train_data
            self.test_data = four_test_data
        else:
            self.train_data = five_train_data
            self.test_data = five_test_data
        self.train = train  # training set or test set

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

        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

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
