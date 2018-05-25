"""Codes for data process"""

import numpy as np
import tensorflow as tf
import pickle
#import random
import os
import logging
import pandas as pd
from collections import defaultdict
import cv2 as cv
import math
import time
import timer
import process_bar

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 32,
                        """batch size for train and test""")
tf.flags.DEFINE_integer('k_fold', 10,
                        """k_cross validation""")

tf.flags.DEFINE_string('ISIC2018_Task3_Training_GroundTruth', '../data/ISIC2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv',
                        """ISIC2018 Task3 Training GroundTruth path""")
tf.flags.DEFINE_string('ISIC2018_Task3_Training_Input', '../data/ISIC2018/ISIC2018_Task3_Training_Input',
                        """ISIC2018 Task3 Training Input path""")
tf.flags.DEFINE_string('ISIC2018', '../data/ISIC2018',
                        """ISIC2018 Task3 path""")


class ISIC2018_data():
    def __init__(self):
        self.task3_training_groundtruth = FLAGS.ISIC2018_Task3_Training_GroundTruth
        self.task3_training_input = FLAGS.ISIC2018_Task3_Training_Input
        self.ISIC2018 = FLAGS.ISIC2018
        self.k_fold = FLAGS.k_fold
        tf.logging.info('ISIC2018_Task3_Training_GroundTruth %s' % self.task3_training_groundtruth)
        tf.logging.info('ISIC2018_Task3_Training_Input %s' % self.task3_training_input)
        self.task3_training_groundtruth_ = pd.read_csv(self.task3_training_groundtruth)
        self.size = self.task3_training_groundtruth_.values.shape[0]
        self.batch_size = FLAGS.batch_size
        self.num_classes = 7
        self.inputs_data = defaultdict(list)
        self.labels_data = defaultdict(list)
        self.labels_name_hard_encode = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        # 10-fold
        # _mean_train, _variance_train means, when index used as validation set, all other train mean and variance
        self._mean_train = []
        self._variance_train = []
        self._mean_valid = []
        self._variance_valid = []
        self.rescale = True # 15
        self.nWid = 400
        self.nHei = 300
        #self.nWid = 224
        #self.nHei = 224
        #self.nWid = 112
        #self.nHei = 112
        #self.nWid = 40
        #self.nHei = 30
        #self.nWid = 120
        #self.nHei = 90
        #self.nWid = 96
        #self.nHei = 96
        self._inputs()

    def set_valid_index(self, i):
        self.val_index= i
        self._load_data_and_norm()

    def _load_data_and_norm(self):
        with open(self.datas_path[self.val_indexi], 'rb') as fo:
            data = pickle.load(fo)
            inputs = np.array(data['inputs'])
            labels = np.array(data['labels'])
            self.inputs_data[self.val_index] = (inputs-np.mean(inputs))/np.std(inputs)
            self.labels_data[self.val_index] = np.array(labels)

        trains_data = []
        for i in range(self.k_fold):
            if i == self.val_index:
                continue
            else:
                with open(self.datas_path[i], 'rb') as fo:
                    data = pickle.load(fo)
                    inputs = np.array(data['inputs'])
                    labels = np.array(data['labels'])
                    self.labels_data[i] = np.array(labels)
                    self.inputs_data[i] = (inputs-np.mean(inputs))/np.std(inputs)
                    trains_data.append(inputs)

        for i in range(self.k_fold):
            if i == self.val_index:
                self.inputs_data[i] = (self.inputs_data[i]-np.mean(trains_data))/np.std(trains_data)


    def get_groups(self, i):
        inputs = self.inputs_data[i]
        labels = self.labels_data[i]

        for x, y in zip(inputs, labels):
            yield x, y


    def _inputs(self):
        data_dir = self.ISIC2018
        output_filename = 'task3_{}_{}_{}'.format(self.batch_size, self.nHei, self.nWid)
        data_dir = os.path.join(data_dir, output_filename)
        tf.logging.info('data_dir %s' % data_dir)
        self.datas_path = []
        for i in range(self.k_fold):
            self.datas_path.append(os.path.join(data_dir, 'task3_norm_%d' %i))
        #if mode == train:
        #    pass
        #elif mode == 'validation':
        #    pass

        #train_batches = np.load(

    def get_shape(self):
        return (self.nHei,self.nWid,3)

    def get_bsize(self):
        return int(math.ceil(self.size/self.k_fold))

    def output_amount_class(self):
        filename = self.task3_training_groundtruth
        self._amount_each_class_from_csv(filename)
        for name in self.imagepath_each_class:
            logging.info('%s : %d' % (name, len(self.imagepath_each_class[name])))

    def generate_inputs_by_batch(self):
        filenames = self.task3_training_groundtruth
        self._amount_each_class_from_csv(filenames)
        groups = self._divide_groups_xy()

        prefix_directory = self.task3_training_input
        index = 0
        img_names = defaultdict(list)
        for filenames_np, labels in groups:
            img_names[str(index)] = filenames_np
            self._generate_batch_by_batch(prefix_directory, filenames_np, labels, index)
            index += 1
        self._generate_each_fold_img_name_as_csv(img_names)

    def _generate_each_fold_img_name_as_csv(self, data):
        """output
            output is a dict
        """
        output_filename = 'each_fold_img_name.csv'
        col_names = [str(i) for i in range(10)]
        output = dict([(k, pd.Series(v)) for k, v in data.items()])
        df = pd.DataFrame(output, columns=col_names)
        with open(output_filename, 'w', newline="") as fo:
            df.to_csv(fo)


    def _generate_batch_by_batch(self, prefix_directory, filenames, labels, index):
        np.random.seed(int(time.time()))
        combine = [(f, l) for f, l in zip(filenames, labels)]
        np.random.shuffle(combine)
        filenames = list(map(lambda x: x[0], combine))
        labels = list(map(lambda x: x[1], combine))
        data = self._load_imgs_by_batch(prefix_directory, filenames)

        groups = int(math.ceil(len(labels)/FLAGS.batch_size))
        labels_ = []
        lens = len(labels)
        for i in range(groups):
            f_ = i*FLAGS.batch_size
            e_ = min((i+1)*FLAGS.batch_size, lens)
            sub_g = labels[f_:e_]
            labels_.append(sub_g)

        process_bar_ = process_bar.process_bar(groups)
            # 10015/64 -> 157
        logging.info('%d has %d batches' % (index, groups))
        batches = []
        for batch_img in data:
            #batch = []
            # Norm
            #for img in batch_img:
                #batch.append(((img-np.mean(img))/np.std(img)))
            # Not Norm TODO
            batch = batch_img
            batches.append(batch)
            process_bar_.show_process()
        assert len(batches) == len(labels_)
        pdata = {'inputs': batches, 'labels': labels_}
        if self.rescale:
            output_filename = 'task3_{}_{}_{}/task3_norm_{}'.format(FLAGS.batch_size, self.nHei, self.nWid,index)
        else:
            output_filename = 'task3_{}/task3_norm_{}'.format(FLAGS.batch_size,index)

        with open(output_filename, 'wb') as fo:
            pickle.dump(pdata, fo)

        #np.save('task3_inputs_norm', batches)
        #np.save('task3_labels', labels)

    def _divide_groups_xy(self):
        label_corresponding = np.zeros((self.num_classes, self.num_classes))
        dia = np.arange(self.num_classes)
        label_corresponding[dia, dia] = 1.0

        xs_path = self.imagepath_each_class
        lens = np.array([len(xs_path[name]) for name in self.labels_name_hard_encode])
        glen = lens//self.k_fold
        for i in range(self.k_fold):
            xs_data = []
            ys_data = []
            for j, name in enumerate(self.labels_name_hard_encode):
                f_ = i*glen[j]
                if i != self.k_fold - 1:
                    e_ = (i+1)*glen[j]
                else:
                    e_ = lens[j]
                xs = xs_path[name][f_:e_]
                ys = [label_corresponding[j]] * (e_-f_)
                assert len(xs) == len(ys)
                xs_data.extend(xs)
                ys_data.extend(ys)
            if i >= 0 and i < 9:
                assert len(xs_data) == 998
            else:
                assert len(xs_data) == 1033
            yield xs_data, ys_data


    def _load_imgs_by_batch(self, prefix_directory, filenames):
        nWid = self.nWid
        nHei = self.nHei
        groups = len(filenames)
        for i in range(int(math.ceil(groups/self.batch_size))):
            f_ = i*self.batch_size
            e_ = min((i+1)*self.batch_size, groups)
            data = []
            for filename in filenames[f_:e_]:
                filename = os.path.join(prefix_directory, filename)
                filename = filename + '.jpg'
                image_np = cv.imread(filename)
                if self.rescale:
                    # wierd!! shape return (nHei, nWid)
                    # resize have the order, (nWid. nHei)
                    image_np= cv.resize(image_np, (nWid, nHei))
                    assert image_np.shape == (nHei, nWid, 3)
                data.append(image_np)
            yield data

    def _amount_each_class_from_csv(self, filename):
        imagepath_each_class = defaultdict(list)
        data = self.task3_training_groundtruth_
        data_np = data.values
        data_column_names = data.columns.values[1:] # skip first columns
        count = 0
        for index, col_names in enumerate(data_column_names, start=1):
            a_class = list(filter(lambda x: x[index] == 1.0, data_np))
            image_names = list(map(lambda x: x[0], a_class))
            imagepath_each_class[col_names].extend(image_names)
            count += len(image_names)
        self.imagepath_each_class = imagepath_each_class
        assert count == 10015


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    timer_ = timer.timer()
    ISIC2018_data_ =  ISIC2018_data()
    ISIC2018_data_.generate_inputs_by_batch()
    #ISIC2018_data_.output_amount_class()
    timer_.get_duration()
