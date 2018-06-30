"""Codes for data process"""
import numpy as np
import tensorflow as tf
#import pickle
import sys
import os
import logging
import pandas as pd
#from collections import defaultdict
import cv2 as cv
#import math
#import time
import process_bar

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 32,
                        """batch size for train and test""")
tf.flags.DEFINE_integer('k_fold', 5,
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
        self.record_outputfilename = 'task3_{}_{}_{}'.format(self.nHei, self.nWid, self.batch_size)
        self.train_dir = 'train_repeat'
        self.records_names = [self.record_outputfilename + '_{}'.format(i) for i in range(self.k_fold)]
        #self._inputs()
        self.extra_init = []
        # set index
        self._set_valid_index(index)

    def _set_valid_index(self, i):
        self.val_index= i
        # train repeat
        self.train_records = [os.path.join(self.record_outputfilename, name) for name in self.records_names if name.split('_')[-1] != str(i)]
        self.train_records = [os.path.join(self.train_dir, i) for i in self.train_records]

        self.train_and_evaluate_record = [os.path.join(self.record_outputfilename, name) for name in self.records_names if name.split('_')[-1] != str(i)]
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
            images = tf.image.resize_images(x, [self.image_size, self.image_size])
        # standardization
        images = tf.image.per_image_standardization(images)
        return images

    def read_record(self, mode='train'):
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
            image = tf.reshape(image, [450, 600, 3])
            image = tf.image.resize_images(image, [self.nHei, self.nWid])

            label = tf.one_hot(label, depth=self.num_classes)

            return image, label, name

        def _preprocess_train(image, label, name):
            # Reshape image data into the original shape
            image = self._pre_process_images(image, is_train=True)
            return {'input_1':image}, label
            #return image, label, name

        def _preprocess_valid(image, label, name):
            # Reshape image data into the original shape
            image = self._pre_process_images(image, is_train=False)
            return {'input_1':image}, label
            #return image, label, name
            #return image, label, name

        def _read_record(path, min_queue_examples=2000, is_train=True, num_epochs=None):
            # parallel for map function
            num_parallel_calls=10
            dataset = tf.data.TFRecordDataset(path)
            dataset = dataset.map(_parse_function, num_parallel_calls=num_parallel_calls)

            # shuffle when is_train is true
            if is_train == True:
                dataset = dataset.map(_preprocess_train, num_parallel_calls=num_parallel_calls)
                num_epochs = self.num_epoch
            else:
                dataset = dataset.map(_preprocess_valid, num_parallel_calls=num_parallel_calls)
                num_epochs = 1

            if is_train == True:
                dataset = dataset.shuffle(buffer_size=min_queue_examples)
                print('shuffle')

            dataset = dataset.repeat(num_epochs)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(buffer_size=FLAGS.batch_size)
            iterator = dataset.make_one_shot_iterator()
            #iterator = dataset.make_initializable_iterator()

            #self.extra_init.append(iterator.initializer)

            # `features` is a dictionary in which each value is a batch of values for
            # that feature; `labels` is a batch of labels.
            features, labels = iterator.get_next()
            #features, labels, names = iterator.get_next()
            return features, labels
            #return features, labels, names
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
            print(self.train_records)
            images, labels = _read_record(self.train_records, is_train=True)
            #images, labels, names = _read_record(self.train_records, is_train=True)
            return images, labels
        elif mode == 'train_evaluation':
            print(self.train_and_evaluate_record)
            images, labels = _read_record(self.train_and_evaluate_record, is_train=False)
            #images, labels, names = _read_record(self.train_records, is_train=False)
            return images, labels
        elif mode == 'evaluate':
            print(self.valid_records)
            images, labels = _read_record(self.valid_records, is_train=False)
            return images, labels

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
            image_np = cv.imread(img_path+'.jpg')
            image_np = cv.cvtColor(image_np, cv.COLOR_BGR2RGB)
            image_np = image_np.astype(np.float32)
            return image_np

        record_outputname = self.record_outputfilename + '_{}'
        record_outputname = record_outputname.format(index)
        tf.logging.info('Generare record %d' % index)

        # write tfrecord
        with tf.python_io.TFRecordWriter(record_outputname) as writer:
            for label, name in zip(labels, images_name):
                img_path = os.path.join(self.task3_training_input, name)
                img = _load_img(img_path).tostring()
                feature = {'label_raw': _int64_feature(int(label)),
                           'image_raw': _bytes_feature(img),
                           'name': _bytes_feature(name.encode('utf-8'))
                           }

                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())

    def write_record(self, prefix_directory=FLAGS.ISIC2018, filenames='split_data.csv'):
        """wrap function to generate record for different record
        """
        def _read_csv(filename):
            """Get filenames, labels from csv
            Args:
                filename
            """
            data = pd.read_csv(filename)
            # get columns images name
            cc = data.columns.values[::2]
            # get columns labels name
            cc_ = data.columns.values[1::2]

            for c, c_ in zip(cc, cc_):
                images_name = list(filter(lambda x: pd.isnull(x) == False, data[c].values))
                labels = list(filter(lambda x: pd.isnull(x) == False, data[c_].values))
                # repeat
                weight_sample = (10015/np.array([1113,6705,514,327,1099,115,142])).astype(np.int64)
                new_images_name = []
                new_labels = []
                for x, y in zip(images_name, labels):
                    new_images_name.extend([x] * weight_sample[int(y)])
                    new_labels.extend([y] * weight_sample[int(y)])
                print(len(new_images_name), len(new_labels))
                yield new_images_name, new_labels
                #yield images_name, labels

        sets = _read_csv(os.path.join(prefix_directory, filenames))
        process_bar_ = process_bar.process_bar(self.k_fold)
        for i in range(self.k_fold):
            images_name, labels = next(sets)
            self._write_record(labels, images_name, i)
        process_bar_.show_process()


def test_record():
    ISIC2018_data_ =  ISIC2018_data(4)
    images, labels, names = ISIC2018_data_.read_record(mode='train_evaluation')

    cnt = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.device('/device:GPU:0'):
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(ISIC2018_data_.extra_init)
            try:
                while True:
                    images_, labels_, names_ =  sess.run([images, labels, names])
                    print(images_.shape, labels_.shape, names_.shape)
                    print(names_)
                    cnt += 1
            except tf.errors.OutOfRangeError:
                print(cnt)


def generate_record():
    ISIC2018_data_ =  ISIC2018_data(4)
    ISIC2018_data_.write_record()


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]
    generate_record()
    #test_record()
