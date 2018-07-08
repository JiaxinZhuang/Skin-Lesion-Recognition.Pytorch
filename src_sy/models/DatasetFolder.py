import csv
import os
import numpy as np
import torch.utils.data as data
from PIL import Image
import random

def getIdx(lblarr, consider_lbl, ptrain):
    index = []
    for i, val in enumerate(lblarr):
        if val==consider_lbl:
            index.append(i)
    train_idx = random.sample(index, int(len(index)*ptrain)) #list(range(1,int(len(index)*ptrain))) #
    test_idx = set(index) - set(train_idx)
    return index, train_idx, test_idx


def make_dataset(ptrain, data_dir):
    images = []
    fnarr = []
    lblarr = []
    #dir = '/home/siyam/DATA/ISIC2018_Task3_Training_Input/'
    #dir = '/home/jiaxin/myGithub/Reverse_CISI_Classification/data/ISIC2018/ISIC2018_Task3_Training_Input/'
    dir_gt = '/home/jiaxin/myGithub/Reverse_CISI_Classification/data/ISIC2018/ISIC2018_Task3_Training_GroundTruth/'
    with open(os.path.join(dir_gt, 'ISIC2018_Task3_Training_GroundTruth.csv'), 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for line_no, row in enumerate(reader):
            if line_no==0: continue
            row = row[0].split(',')
            name = os.path.join(data_dir, row[0]+'.jpg')
            row = row[1:]
            for i,val in enumerate(row):
                if int(float(val))==1:
                    item = (name, i)
                    images.append(item)
                    fnarr.append(name)
                    lblarr.append(i)


    train_data = []
    test_data = []
    unique_labels = np.unique(lblarr).tolist()
    count = []
    for lbl in unique_labels:
        index, train_idx, test_idx = getIdx(lblarr, lbl, ptrain)
        if len(list(set(train_idx).intersection(test_idx)))!=0:
            raise ValueError('Problems with overlapping train and test split!')

        print(len(index))
        count.append(len(index))
        for i in train_idx:
            train_data.append(images[i])
        for i in test_idx:
            test_data.append(images[i])
        print(f"Number of images in class  {lbl} is  {len(index)} : {len(train_idx)}, {len(test_idx)}")
    print(f'Total number of images loaded : {len(images)} : {len(train_data)}, {len(test_data)}')

    sv = sum(count)
    weights = []
    tot = 0;
    for i, val in enumerate(count):
        weights.append((sv-val)/val)
        tot = tot + weights[i]
    for val in weights:
       print(np.round(val/tot,3), end=',')
    print()

    for val in count:
       print(val, end=',')
    print()


#    for item in train_data:
#        print(item[0], ',', item[1])
    return train_data, test_data


class DatasetFolder(data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, iterNo=1, data_dir=None):
        self.train_data, self.test_data = make_dataset(iterNo, data_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train # training set or test set


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
        if self.target_transform is not None:
            target = self.target_transform(target)

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

#make_dataset(0.5)

