"""using vgg16 to extract feature"""

import tensorflow as tf
from tensorflow.python.keras.applications import resnet50
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras.optimizers import SGD
from keras import metrics
import numpy as np
from collections import namedtuple

import time
import os, sys
import data_utils
parent_path = os.path.abspath('../')
sys.path.insert(0, parent_path)

import statistics

def main(mode='train', gpu='0'):
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu

    data = data_utils.ISIC2018_data(4)
    num_classes = 7

    # extract feature
    base_model = resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))

    images_feature = []
    y_list = []
    with tf.Session() as sess:
        try:
            if mode == 'train_evaluation':
                x, y = data.read_record('train_evaluation')
                outputname = 'images_feature_with_labels_from_vgg16_train'
            elif mode == 'evaluate':
                x, y = data.read_record('evaluate')
                outputname = 'images_feature_with_labels_from_vgg16_evaluate'

            sess.run(tf.global_variables_initializer())
            cnt = 0
            while True:
                x_, y_ = sess.run([x, y])
                features = base_model.predict(x_)
                y_list.append((np.argmax(y_, axis=1)))
                #print(features.shape)
                images_feature.append(features)
                cnt += 1
                print('batch %d' % cnt)
        except tf.errors.OutOfRangeError:
            print('load finish labels')

    x_y = [images_feature, y_list]
    np.save(outputname, x_y)


if __name__ == '__main__':
    args = sys.argv[1]
    gpu = sys.argv[2]
    #main(mode='evaluate')
    main(mode=args, gpu=gpu)
