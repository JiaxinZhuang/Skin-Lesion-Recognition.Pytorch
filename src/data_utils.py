"""Codes for data process"""
import numpy as np
import tensorflow as tf
import pickle
import sys
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
    def __init__(self, index, num_epoch=None):
        self.num_epoch = 250
        self.task3_training_groundtruth = FLAGS.ISIC2018_Task3_Training_GroundTruth
        self.task3_training_input = FLAGS.ISIC2018_Task3_Training_Input
        self.ISIC2018 = FLAGS.ISIC2018
        self.k_fold = FLAGS.k_fold
        tf.logging.info('ISIC2018_Task3_Training_GroundTruth %s' % self.task3_training_groundtruth)
        tf.logging.info('ISIC2018_Task3_Training_Input %s' % self.task3_training_input)
        self.task3_training_groundtruth_ = pd.read_csv(self.task3_training_groundtruth)
        self.split_filename = os.path.join(self.ISIC2018, 'split_{}'.format(self.k_fold))
        self.size = self.task3_training_groundtruth_.values.shape[0]
        self.batch_size = FLAGS.batch_size
        self.num_classes = 7
        #self.inputs = defaultdict(list)
        self.inputs_data = []
        self.labels_data = []
        self.labels_name_hard_encode = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        # 10-fold
        # image size is for model
        self.image_size = 224
        self.rescale = True # 15
        self.nWid = 400
        self.nHei = 300
        self.record_outputfilename = 'task3_{}_{}_{}'.format(self.nHei, self.nWidth, self.batch_size)
        self.records_names = [self.record_outputfilename + '_{}'.format(i) for i in range(self.k_fold)]
        self._inputs()
        self.extra_init = []
        # set index
        self._set_valid_index(index)

    def _set_valid_index(self, i):
        self.val_index= i
        self.train_records = [os.path.join(self.record_outputfilename, name) for name in self.records_names if name.split('_')[-1] != str(i)]
        self.valid_records = [os.path.join(self.record_outputfilename, self.record_outputfilename+'_{}'.format(i))]

    def _pre_process_images(self, x, is_train=True):
        """
        Inputs:
            x: [300, 400, 3]
        Returns:
               [224, 224, 3]
        """
        if is_train:
            # resize
            images = tf.image.resize_image_with_crop_or_pad(x, self.image_size+4, self.image_size+4)
            # random crop
            images = tf.random_crop(images, [self.image_size, self.image_size, 3])
            # flip
            images = tf.image.random_flip_left_right(images)
            images = tf.image.random_flip_up_down(images)
        else:
            # resize
            images = tf.image.resize_image_with_crop_or_pad(x, self.image_size, self.image_size)
        # standardization
        images = tf.image.per_image_standardization(images)
        return images

    def load_record(self, mode='train'):
        def _parse_function(example_proto):
            feature = {'image_raw': tf.FixedLenFeature([], tf.string),
                       'label_raw': tf.FixedLenFeature([], tf.int64),
                       'name': tf.FixedLenFeature([], tf.string)
                       }
            parsed_features = tf.parse_single_example(example_proto, features=feature)
            image = parsed_features['image_raw']
            label = parsed_features['label_raw']
            name = parsed_features['name']

            image = tf.decode_raw(image, tf.float32)
            image = tf.reshape(image, [self.nHei, self.nWid, 3])

            label = tf.one_hot(label, depth=self.num_classes)

            return image, label, name

        def _preprocess_train(image, label):
            # Reshape image data into the original shape
            image = self._pre_process_images(image, is_train=True)
            return image, label

        def _preprocess_valid(image, label):
            # Reshape image data into the original shape
            image = self._pre_process_images(image, is_train=False)
            return image, label

        def _load_record(path, min_queue_examples=2000, is_train=True, num_epochs=None):
            # parallel for map function
            num_parallel_calls=10
            filename_queue = tf.train.string_input_producer(path)
            dataset = tf.data.TFRecordDataset(filename_queue)
            dataset = dataset.map(_parse_function, num_parallel_calls=num_parallel_calls)

            # shuffle when is_train is true
            if is_train == True:
                dataset = dataset.map(_preprocess_train, num_parallel_calls=num_parallel_calls)
            else:
                assert num_epochs != None
                dataset = dataset.map(_preprocess_valid, num_parallel_calls=num_parallel_calls)

            dataset = dataset.shuffle(buffer_size=min_queue_examples)
            dataset = dataset.repeat(num_epochs)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(buffer_size=FLAGS.batch_size)
            iterator = dataset.make_initializable_iterator()

            self.extra_init.append(iterator.initializer)

            # `features` is a dictionary in which each value is a batch of values for
            # that feature; `labels` is a batch of labels.
            features, labels = iterator.get_next()
            return features, labels
            #num_preprocess_threads = 8

            #if is_train:
            #    images, labels = tf.train.shuffle_batch([image, label],
            #            batch_size=FLAGS.batch_size,
            #            num_threads=num_preprocess_threads,
            #            min_after_dequeue=min_queue_examples,
            #            capacity=min_queue_examples+FLAGS.batch_size * 3,
            #            allow_smaller_final_batch=True)
            #else:
            #    images, labels = tf.train.batch([image, label],
            #            batch_size=FLAGS.batch_size,
            #            num_threads=num_preprocess_threads,
            #            capacity=min_queue_examples+FLAGS.batch_size * 3,
            #            allow_smaller_final_batch=True)

        if mode == 'train':
            images, labels, names = _load_record(self.train_records, is_train=True)
        elif mode == 'train_evaluation':
            images, labels, names = _load_record(self.train_records, is_train=False)
        elif mode == 'evaluate':
            images, labels = _load_record(self.valid_records, is_train=False)
        return images, labels, names

    def get_inputs(self):
        return self.inputs_data, self.labels_data

    def get_train(self):
        return self.train_images, self.train_labels

    def get_valid(self):
        return self.valid_images, self.valid_labels

    def get_origin_shape(self):
        return (self.nHei, self.nWid, 3)

    def get_shape(self):
        return (self.image_size,self.image_size,3)

    def _generate_each_fold_img_name_as_csv(self, data):
        """output
            output is a dict
        """
        output_filename = 'each_fold_img_name.csv'
        col_names = [str(i) for i in range(10)]
        output = dict([(k, pd.Series(v)) for k, v in data.items()])
        df = pd.DataFrame(output, columns=col_names)
        with open(output_filename, 'w', newline="") as fo:
            df.to_csv(fo, index=False)


    def _write_record(self, labels, images_name, index):
        """write record
        Args
            images_path: a list, path to images for loading
            labels_path: a list, labels, a real number
            images_name: a list, image name
            index: outputfile index
        """
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        def _load_img(img_path):
            image_np = cv.imread(filename)
            image_np = cv.cvtColor(image_np, cv.COLOR_BGR2RGB)
            image_np = image_np.astype(np.float32)
            return image_np

        record_outputname = self.record_outputfilename + '_{}'
        record_outputname = recood_outputfilename.format(index)
        tf.logging.info('Generare record %d' % index)

        # write tfrecord
        with tf.python_io.TFRecordWriter(output_filename) as writer:
            for label, name in zip(labels, images_name):
                img_path = os.path.join(self.task3_training_input, name)
                img = _load_img(img_path).tostring()
                feature = {'label_raw': _int64_feature(label),
                           'image_raw': _bytes_feature(img),
                           'name': _bytes_feature(path)
                           }

                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())

    def generate_record(self, prefix_directory, filenames, labels):
        """wrap function to generate record for different record
        """
        def _read_csv(filename):
            images_name_dict = defaultdict(list)
            labels_path_dict = defaultdict(list)

        sets = pd.read_csv(filename)
        process_bar_ = process_bar.process_bar(self.k_fold)
        for i in range(self.k_fold):
            names = sets.loc('%d_images_name'.format(i))
            labels = sets.loc('%d_labels'.format)
            self._write_record(labels, names, i)
        process_bar_.show_process()


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    timer_ = timer.timer()
    # generate data batch by batch  using tfrecord
    # including training dataset and validation set
    ISIC2018_data_ =  ISIC2018_data()
    ISIC2018_data_.generate_inputs_by_batch()
    timer_.get_duration()
    #ISIC2018_data_.set_valid_index(4)
    #for i in range(ISIC2018_data_.k_fold):
    #    data = ISIC2018_data_.get_groups(i)
    #    for x, y in data:
    #        print(np.array(x).shape)
    #        print(np.array(y).shape)
    #data = defaultdict(list)
    #process_bar_ = process_bar.process_bar(ISIC2018_data_.k_fold)
    #i = 0
    #for val, tra in get_mean_std:
    #    print(val)
    #    print(tra)

    #    data[str(i)+'_train'] = tra
    #    data[str(i)+'_val'] = val
    #    i += 1
    #    #process_bar_.show_process()

    # output amount for each class
    #ISIC2018_data_.output_amount_class()

    # generate train and set set according to csv files
    #ISIC2018_data_.generate_train_valid()
