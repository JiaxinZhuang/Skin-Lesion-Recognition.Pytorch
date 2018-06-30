import pandas as pd
import numpy as np
import os
from shutil import copyfile


def get_set(filename):
    data = pd.read_csv(filename)
    # get columns images name
    cc = data.columns.values[::2]
    # get columns labels name
    cc_ = data.columns.values[1::2]

    cnt = 0
    for c, c_ in zip(cc, cc_):
        images_name = list(filter(lambda x: pd.isnull(x) == False, data[c].values))
        labels = list(filter(lambda x: pd.isnull(x) == False, data[c_].values))
        weight_sample = (10015/np.array([1113,6705,514,327,1099,115,142])).astype(np.int64)
        if cnt != 4:
            new_images_name  = None
            new_labels = None
            pass
            # repeat
            #new_images_name = []
            #new_labels = []
            #for x, y in zip(images_name, labels):
            #    temp = []
            #    for i in range(weight_sample[int(y)]):
            #        temp.append(x+'_'+str(i))
            #    new_images_name.extend(temp)
            #    new_labels.extend([int(y)] * weight_sample[int(y)])
            #print(len(new_images_name), len(new_labels))
        else:
            new_images_name = images_name
            new_labels = labels
        yield new_images_name, new_labels
        cnt += 1
        #yield images_name, labels

sets = get_set('../data/ISIC2018/split_data.csv')
k_fold=5

def cp_file(images_name, labels, train=True):
    src = '../data/ISIC2018/ISIC2018_Task3_Training_Input/'
    dst_train = '../data/task3/train'
    dst_val = '../data/task3/val'

    if train:
        dst = dst_train
    else:
        dst = dst_val

    for x, y in zip(images_name, labels):
        if train:
            x_ = x.rsplit('_', maxsplit=1)[0]
        else:
            x_ = x
        src_file = os.path.join(src, x_)
        src_file += '.jpg'
        dst_file = os.path.join(dst, str(int(y)))
        dst_file = os.path.join(dst_file, x)
        dst_file += '.jpg'
        copyfile(src_file, dst_file)
        print(src_file, dst_file)


for i in range(k_fold):
    images_name, labels = next(sets)
    if i != 4:
        pass
        #cp_file(images_name, labels, train=True)
    else:
        cp_file(images_name, labels, train=False)

