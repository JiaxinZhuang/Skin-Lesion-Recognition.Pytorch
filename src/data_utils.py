"""Codes for data process"""
import numpy as np
import tensorflow as tf
import pickle
#import random
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
    def __init__(self):
        self.epoch = 250
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
        #self.inputs = defaultdict(list)
        self.inputs_data = []
        self.labels_data = []
        self.labels_name_hard_encode = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        #self.mean_val_tra_path = 'mean_val_tra.csv'
        # 10-fold
        # _mean_train, _variance_train means, when index used as validation set, all other train mean and variance
        #self._mean_train = []
        #self._variance_train = []
        #self._mean_valid = []
        #self._variance_valid = []
        # image size is for model
        self.image_size = 224
        self.rescale = True # 15
        self.nWid = 400
        self.nHei = 300
        #self.nWid = 40
        #self.nHei = 30
        self._inputs()
        self.extra_init = []

    def generate_train_valid(self):
        train_path = '../data/ISIC2018/2018_5_28/train_label_list.csv'
        train_path_ = '../data/ISIC2018/2018_5_28/train_label_list_.csv'
        test_path = '../data/ISIC2018/2018_5_28/test_label_list.csv'
        test_path_ = '../data/ISIC2018/2018_5_28/test_label_list_.csv'
        self.output_amount_class()

        def _generate(files, ofiles):
            train_set = pd.read_csv(files, header=None).values[:,1]
            assert train_set.shape[0] == 8010 or train_set.shape[0] == 2005
            #train_label = []

            #label_corresponding = np.zeros((self.num_classes, self.num_classes))
            #dia = np.arange(self.num_classes)
            #label_corresponding[dia, dia] = 1.0

            output = []
            data_np = self.task3_training_groundtruth_.values
            for x in train_set:
                for y in data_np:
                    if y[0] == x:
                        #train_label.append(j)
                        output.append(y)
                        break


            assert train_set.shape[0] == len(output)
            #col_names = {'inputs', 'labels'}
            #output = {'inputs': train_set, 'labels': train_label}
            df = pd.DataFrame(output)
            col_names = ['image']
            col_names.extend(self.labels_name_hard_encode)
            df.columns = col_names
            with open(ofiles, 'w', newline="") as fo:
                df.to_csv(fo, index=False)

        _generate(train_path, train_path_)
        _generate(test_path, test_path_)


    def set_valid_index(self, i):
        self.val_index= i
        self._load_record()

        # load from pickle and norm
        #self._load_data_and_norm(norm)

    def _pre_process_images(self, x, is_train=True):
        """
        Inputs:
            x: [300, 400, 3]
        Returns:
               [224, 224, 3]
        """
        # random crop
        #images = tf.map_fn(lambda img: tf.random_crop(img, [self.image_size, self.image_size, 3]), images)
        # flip
        #images = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), images)
        #images = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), images)
        # standaedization
        #images = tf.map_fn(lambda img: tf.image.per_image_standardization(img), images)

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


    def _load_record(self):
        # Train
        def _parse_function(example_proto):
            feature = {'train/image': tf.FixedLenFeature([], tf.string),
                       'train/label': tf.FixedLenFeature([], tf.int64)}
            parsed_features = tf.parse_single_example(example_proto, features=feature)
            image = parsed_features['train/image']
            label = parsed_features['train/label']

            image = tf.decode_raw(image, tf.float32)
            image = tf.reshape(image, [self.nHei, self.nWid, 3])

            label = tf.one_hot(label, depth=self.num_classes)

            return image, label

        def _preprocess_train(image, label):
            # Reshape image data into the original shape
            image = self._pre_process_images(image, is_train=True)
            return image, label

        def _preprocess_valid(image, label):
            # Reshape image data into the original shape
            image = self._pre_process_images(image, is_train=False)
            return image, label

        def _load_record(path, min_queue_examples, is_train=True, num_epochs=None):
            # epoch is defined in train
            num_parallel_calls=10
            #filename_queue = tf.train.string_input_producer(path, shuffle=False)
            filename_queue = path
            dataset = tf.data.TFRecordDataset(filename_queue)
            dataset = dataset.map(_parse_function, num_parallel_calls=num_parallel_calls)

            if is_train == True:
                num_epochs = 1 if num_epochs == None else num_epochs
                assert num_epochs != None
                dataset = dataset.map(_preprocess_train, num_parallel_calls=num_parallel_calls)
            else:
                num_epochs = 1 if num_epochs == None else num_epochs
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

        min_queue_examples = int(8000 * 0.4)
        train_dir = []
        val_dir = []
        for i, path in enumerate(self.datas_path):
            if i == self.val_index:
                val_dir.append(path)
            else:
                train_dir.append(path)
        images, labels = _load_record(train_dir, min_queue_examples, is_train=True)
        self.inputs_data = images
        self.labels_data = labels
        images, labels = _load_record([self.train_path], min_queue_examples, is_train=False)
        self.train_images = images
        self.train_labels = labels
        images, labels = _load_record(val_dir, min_queue_examples, is_train=False)
        self.valid_images = images
        self.valid_labels = labels

    def get_inputs(self):
        return self.inputs_data, self.labels_data

    def get_train(self):
        return self.train_images, self.train_labels

    def get_valid(self):
        return self.valid_images, self.valid_labels

    def _load_data_and_norm(self, norm=True):
        output_filename = '2018_5_28'
        output_filename = os.path.join(self.ISIC2018, output_filename)
        output_filename = os.path.join(output_filename, self.mean_val_tra_path)
        if norm:
            if not os.path.exists(output_filename):
                print('Give mean std file')
                sys.exit(-1)
            data = pd.read_csv(output_filename)
            mean_for_val = data[str(self.val_index)+'_train'].values
            mean_for_tra = data[str(self.val_index)+'_val'].values
            assert mean_for_tra.shape[0] == 3
            assert mean_for_val.shape[0] == 3
            print(mean_for_val)
            #print(std_for_val)
            print(mean_for_tra)
            #print(std_for_tra)

        with open(self.datas_path[self.val_index], 'rb') as fo:
            data = pickle.load(fo)
            inputs = np.array(data['inputs'])
            labels = np.array(data['labels'])

            inputs_ = []
            if norm:
                for x in inputs:
                    inputs_.append(np.array(x-mean_for_val))
                #inputs_ = (inputs-mean_for_val)/std_for_val
            else:
                for x in inputs:
                    inputs_.append(np.array(x))

            labels_ = []
            for label in labels:
                labels_.append(np.array(label))

            self.inputs_data[self.val_index] = inputs_
            self.labels_data[self.val_index] = labels_


        for i in range(self.k_fold):
            if i == self.val_index:
                continue
            else:
                with open(self.datas_path[i], 'rb') as fo:
                    data = pickle.load(fo)
                    inputs = np.array(data['inputs'])
                    labels = np.array(data['labels'])

                    labels_ = []
                    for label in labels:
                        labels_.append(np.array(label))
                    self.labels_data[i] = labels_

                    inputs_ = []
                    if norm:
                        for x in inputs:
                            inputs_.append(np.array(x-mean_for_tra))
                    else:
                        for x in inputs:
                            inputs_.append(np.array(x))

                    self.inputs_data[i] = inputs_


    def get_groups(self, i):
        inputs = self.inputs_data[i]
        labels = self.labels_data[i]

        #inputs, labels = inputs_.make_one_shot_iterator()
        return inputs, labels
        #for x, y in zip(inputs, labels):
            #x = self._pre_process_images(x)
            #yield x, y

    #def _pre_process_images(self, x):
    #    output = []
    #    for xx in x:
    #        image = image = tf.image.resize_image_with_crop_or_pad(
    #                        xx, self.image_size+4 , self.image_size+4)
    #        image = tf.random_crop(image, [self.image_size, self.image_size, 3])
    #        image = tf.image.random_flip_left_right(image)
    #        output.append(image)
    #    return np.array(output)

    def _inputs(self):
        data_dir = self.ISIC2018
        # Read filename
        output_filename = '2018_6_4/task3_{}_{}_{}'.format(self.batch_size, self.nHei, self.nWid)
        #output_filename = '2018_5_28/task3_{}_{}_{}'.format(self.batch_size, self.nHei, self.nWid)
        data_dir = os.path.join(data_dir, output_filename)
        tf.logging.info('data_dir %s' % data_dir)
        self.datas_path = []
        for i in range(self.k_fold):
            self.datas_path.append(os.path.join(data_dir, 'task3_norm_%d' %i))
        self.train_path = os.path.join(data_dir, 'train')
        #if mode == train:
        #    pass
        #elif mode == 'validation':
        #    pass

        #train_batches = np.load(

    def get_origin_shape(self):
        return (self.nHei, self.nWid, 3)

    def get_shape(self):
        return (self.image_size,self.image_size,3)

    def get_bsize(self):
        return int(math.ceil(self.size/self.k_fold))

    def output_amount_class(self):
        filename = self.task3_training_groundtruth
        self._amount_each_class_from_csv(filename)
        for name in self.imagepath_each_class:
            logging.info('%s : %d' % (name, len(self.imagepath_each_class[name])))

    def _scale_amount_for_fewer(self):
        """scale images name for fewer
        """
        weight_sample = 6705/np.array([1113,6705,514,327,1099,115,142])
        weight_sample = weight_sample.astype(int)
        for index, name in enumerate(self.labels_name_hard_encode):
            self.imagepath_each_class[name] = self.imagepath_each_class[name] * weight_sample[index]
            print(len(self.imagepath_each_class[name]))

    def generate_inputs_by_batch(self):
        #self.output_amount_class()
        # scale only when training
        filenames = self.task3_training_groundtruth
        self._amount_each_class_from_csv(filenames)
        self._scale_amount_for_fewer()
        #sys.exit(-1)
        groups = self._divide_groups_xy()

        prefix_directory = self.task3_training_input
        index = 0
        img_names = defaultdict(list)
        for filenames_np, labels in groups:
            img_names[str(index)] = filenames_np
            self._generate_record(prefix_directory, filenames_np, labels, index)
            #self._generate_batch_by_batch(prefix_directory, filenames_np, labels, index)
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
            df.to_csv(fo, index=False)



    def _generate_record(self, prefix_directory, filenames, labels, index):
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        output_filename = 'task3_{}_{}_{}/task3_norm_{}'.format(FLAGS.batch_size, self.nHei, self.nWid,index)
        data = self._load_imgs_by_batch(prefix_directory, filenames)
        groups = int(math.ceil(len(labels)/FLAGS.batch_size))
        labels_ = []
        lens = len(labels)
        for i in range(groups):
            f_ = i*FLAGS.batch_size
            e_ = min((i+1)*FLAGS.batch_size, lens)
            sub_g = labels[f_:e_]
            labels_.append(sub_g)


        process_bar_ = process_bar.process_bar(lens)
        #writer = tf.python_io.TFRecordWriter(output_filename)
        with tf.python_io.TFRecordWriter(output_filename) as writer:
            for index, batch_img in enumerate(data):
                for img, label in zip(batch_img, labels_[index]):
                    feature = {'train/label': _int64_feature(label),
                               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

                    example = tf.train.Example(features=tf.train.Features(feature=feature))

                    # Serialize to string and write on the file
                    writer.write(example.SerializeToString())

                    process_bar_.show_process()


    def _generate_batch_by_batch(self, prefix_directory, filenames, labels, index):
        """generate batch by batch to in pickle file
        """
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
        pdata = {'inputs': np.array(batches), 'labels': labels_}
        if self.rescale:
            output_filename = 'task3_{}_{}_{}/task3_norm_{}'.format(FLAGS.batch_size, self.nHei, self.nWid,index)
        else:
            output_filename = 'task3_{}/task3_norm_{}'.format(FLAGS.batch_size,index)

        with open(output_filename, 'wb') as fo:
            pickle.dump(pdata, fo)

        #np.save('task3_inputs_norm', batches)
        #np.save('task3_labels', labels)

    def _divide_groups_xy(self):
        #label_corresponding = np.zeros((self.num_classes, self.num_classes))
        #dia = np.arange(self.num_classes)
        #label_corresponding[dia, dia] = 1.0

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
                #ys = [label_corresponding[j]] * (e_-f_)
                ys = [j] * (e_-f_)
                assert len(xs) == len(ys)
                xs_data.extend(xs)
                ys_data.extend(ys)
            #if i >= 0 and i < 9:
            #    assert len(xs_data) == 998
            #else:
            #    assert len(xs_data) == 1033
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
                image_np = cv.cvtColor(image_np, cv.COLOR_BGR2RGB)
                image_np = image_np.astype(np.float32)
                if self.rescale:
                    # wierd!! shape return (nHei, nWid)
                    # resize have the order, (nWid. nHei)
                    image_np= cv.resize(image_np, (nWid, nHei))
                    assert image_np.shape == (nHei, nWid, 3)
                data.append(image_np)
            yield np.array(data)

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
        #assert count == 10015

    def _compute_mean_std_for_val_index_others(self):
        """
            Inputs:
                i: val_index
            return:
                (mean_for_val, std_for_val, mean_for_others, std_for_others)
        """
        for i in range(self.k_fold):
            with open(self.datas_path[i], 'rb') as fo:
                data = pickle.load(fo)
                self.inputs_data[i] = data['inputs']

        tf.logging.info('Finishing Loading')

        for i in range(self.k_fold):
            tf.logging.info('Val_index is %d' % i)
            vals_data = []
            for x in self.inputs_data[i]:
                vals_data.extend(x)
            mean_for_val = np.mean(vals_data, axis=(0,1,2))
            assert mean_for_val.shape[0] == 3
            #std_for_val = np.std(vals_data)
            tf.logging.info(mean_for_val)
            #tf.logging.info(std_for_val)

            trains_data = []
            for j in range(self.k_fold):
                if i == j:
                    continue
                else:
                    for x in self.inputs_data[j]:
                        trains_data.extend(x)
            mean_for_tra = np.mean(trains_data, axis=(0,1,2))
            assert mean_for_tra.shape[0] == 3

            #std_for_tra = np.std(trains_data)
            tf.logging.info(mean_for_tra)
            #tf.logging.info(std_for_tra)
            #yield mean_for_val, std_for_val, mean_for_tra, std_for_tra
            yield mean_for_val, mean_for_tra


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    timer_ = timer.timer()
    ISIC2018_data_ =  ISIC2018_data()
    #ISIC2018_data_.set_valid_index(4)
    #for i in range(ISIC2018_data_.k_fold):
    #    data = ISIC2018_data_.get_groups(i)
    #    for x, y in data:
    #        print(np.array(x).shape)
    #        print(np.array(y).shape)
    #get_mean_std = ISIC2018_data_._compute_mean_std_for_val_index_others()
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

    #df = pd.DataFrame(data)
    ##df.columns = col_names
    #with open('mean_val_tra.csv', 'w', newline="") as fo:
    #    df.to_csv(fo, index=False)

    ISIC2018_data_.generate_inputs_by_batch()

    # output amount for each class
    #ISIC2018_data_.output_amount_class()

    # generate train and set set according to csv files
    #ISIC2018_data_.generate_train_valid()
    timer_.get_duration()
