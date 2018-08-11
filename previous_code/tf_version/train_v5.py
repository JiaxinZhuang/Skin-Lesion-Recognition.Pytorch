"""Using ResNet50 or ResNet152 from keras to implement"""

import tensorflow as tf

from tensorflow.python.keras.applications import resnet50
#from keras.applications import densenet
#from tensorflow.python.keras.applications import densenet
from tensorflow.python.keras.applications import inception_v3
#import resnet152
import resnet

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

HParams = namedtuple('HParams',
                     'num_classes')
hps = HParams(
     num_classes=7)
     #weight_sample=weight_sample_,

def main(mode='train', gpu='0'):
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu

    data = data_utils.ISIC2018_data(4)
    num_classes = 7

    # using resnet50
    #base_model = resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))

    # using resnet152
    #base_model = resnet152.ResNet152(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))

    # using resnet152
    #base_model = resnet.ResNet152(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))

    # using densenet
    #base_model = densenet.DenseNet201(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))

    # inception v3
    base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))

    model = models.Sequential()

    model.add(base_model)
    # remove fully connected according to paper
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax', name='fc7'))
    base_model.trainable = False
    #model.compile(loss=median_weight_class_loss,
    #model.compile(loss='categorical_crossentropy',
    model.compile(loss=focal_loss,
                  optimizer=SGD(lr=0.001, momentum=0.9, decay=0.0),
                  metrics=[metrics.categorical_accuracy])

    model_dir = '/home/jiaxin/myGithub/Reverse_CISI_Classification/src/inception_v3_keras_pre/model_fc_3'
    #model_dir = '/home/jiaxin/myGithub/Reverse_CISI_Classification/src/resnet152_keras_pre/model_fc'
    #model_dir = '/home/jiaxin/myGithub/Reverse_CISI_Classification/src/resnet152_keras_pre/model_fc'
    #model_dir = '/home/jiaxin/myGithub/Reverse_CISI_Classification/src/resnet50_keras_pre/model_cw'
    #model_dir = '/home/jiaxin/myGithub/Reverse_CISI_Classification/src/resnet50_keras_pre/model_fc_cw'
    #model_dir = '/home/jiaxin/myGithub/Reverse_CISI_Classification/src/resnet50_keras_pre/model_fc_cw_nor'
    os.makedirs(model_dir, exist_ok=True)
    print('model_dir', model_dir)
    est = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                model_dir=model_dir)

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: data.read_record('train'))
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: data.read_record('train_evaluation'))

    if mode == 'train':
        tf.estimator.train_and_evaluate(est, train_spec, eval_spec)
    elif mode == 'evaluate':
        with tf.Session() as sess:
            try:
                x, y = data.read_record('evaluate')
                sess.run(tf.global_variables_initializer())
                y_list = []
                while True:
                    _, y_ = sess.run([x, y])
                    y_list.extend((np.argmax(y_, axis=1)))
            except tf.errors.OutOfRangeError:
                print('load finish labels')

            # test
            #cnt = 0
            #while True:
            #    try:
            #        x, y = data.read_record('evaluate')
            #        sess.run(tf.global_variables_initializer())
            #        y_list_ = []
            #        while True:
            #            _, y_ = sess.run([x, y])
            #            y_list_.extend((np.argmax(y_, axis=1)))
            #    except tf.errors.OutOfRangeError:
            #        cnt += 1
            #        print(cnt)
            #        assert all([a==b for a, b in zip(y_list, y_list_)])

            pp = []
            while True:
                predictions = est.predict(input_fn=lambda: data.read_record('evaluate'))
                predictions_list = []
                for pre in predictions:
                    p = np.argmax(pre['fc7'])
                    predictions_list.append(p)

                statistics_ = statistics.statistics(hps, mode='evaluate')
                statistics_.add_labels_predictions(predictions_list, y_list)
                statistics_.get_acc_normal()
                result = statistics_.get_acc_imbalanced()
                np.save('predictions_label_fc_3', [predictions_list, y_list])
                #np.save('predictions_label_fc_without_fulcon', [predictions_list, y_list])
                pp.append(result)

                print('---')
                np.save('result_fc_3', pp)
                #np.save('result_fc_without_fulcon', pp)
                time.sleep(120)

def focal_loss(labels, logits):
    gamma=2.0
    alpha=4.0
    epsilon = tf.constant(value=1e-9)
    softmax = tf.nn.softmax(logits)
    model_out = tf.add(softmax, epsilon)
    ce = tf.multiply(labels, -tf.log(model_out))

    # FC
    weight = tf.multiply(labels, tf.pow(tf.subtract(1., model_out), gamma))

    # mask 1 class keep it weight as origin 1
    mask = tf.constant([0.0,1.0,0.0,0.0,0.0,0.0,0.0])
    class_1 = tf.multiply(mask, weight)
    weight = tf.add(-class_1+1, weight)

    # multiply median fre
    #weight_sample = np.array([1113,6705,514,327,1099,115,142])/10015
    #weight_sample = 0.05132302/weight_sample
    #weight = tf.multiply(weight, weight_sample)


    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    #reduced_fl = tf.reduce_sum(fl, axis=1, keep_dims=True)
    reduced_fl = tf.reduce_sum(fl, axis=1)
    return reduced_fl

def median_weight_class_loss(labels, logits):
    epsilon = tf.constant(value=1e-10)
    logits = logits + epsilon
    softmax = tf.nn.softmax(logits)
    weight_sample = np.array([1113,6705,514,327,1099,115,142])/10015
    weight_sample = 0.05132302/weight_sample
    xent = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), weight_sample), axis=1)
    return xent

if __name__ == '__main__':
    #main(mode='evaluate')
    main(mode=sys.argv[1], gpu=sys.argv[2])
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#x = Dense(1024, activation='relu')(x)
#predictions = Dense(num_classes, activation='softmax')(x)
#model = Model(inputs=base_model.input, outputs=predictions)

#for layer in base_model.layers:
#    layer.trainable = False
#
#model.compile(loss='categorical_crossentropy',
#x_train = resnet50.preprocess_input(x_train)
#
#    print(model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0))
#        model.fit(x_train, y_train,
#                          epochs=100,
#                                    batch_size=batch_size,
#                                              shuffle=False,
#                                                        validation_dat=(x_train, y_train))
