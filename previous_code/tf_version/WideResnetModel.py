import tensorflow as tf
import numpy as np

from collections import namedtuple
from tensorflow.python.training import moving_averages
HParams = namedtuple('HParams',
                     'batch_size, num_classes, '
                     'conv2_nums, conv3_nums, conv4_nums, conv5_nums,'
                     'lrn_rate, weight_decay_rate')

class WideResNet(object):
    """
    ResNet model.
    """

    def __init__(self, hps, images, labels, imgids, mode):
        self.hps = hps
        self.images = images
        self.uint8imgs = tf.cast(images, tf.uint8)
        self.labels = tf.one_hot(labels, depth=hps.num_classes, axis=-1)
        self.imgids = imgids

        self.mode = mode

        self._extra_train_ops = []

    def build_graph(self):
        self.global_step = tf.train.get_or_create_global_step()
        self._build_model()
        if self.mode == 'train':
            self._build_train_op()
        self.summaries = tf.summary.merge_all()

    def _stride_arr(self, stride):
        return [1, stride, stride, 1]

    def _decay(self):
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters],
                                     tf.float32, initializer=tf.random_normal_initializer(
                                        stddev=np.sqrt(2.0/n)))
            return tf.nn.conv2d(x, kernel, strides=strides, padding='SAME')

    def _relu(self, x):
        return tf.nn.relu(x, name='relu')

    def _fully_connected(self, x, out_dim):
        x = tf.reshape(x, [self.hps.batch_size, -1])
        w = tf.get_variable('DW', [x.get_shape()[1], out_dim],
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    def _batch_norm(self, name, x):
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable('beta', params_shape, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma', params_shape, tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                                              initializer=tf.constant_initializer(0.0, tf.float32),
                                              trainable=False)
                moving_variance = tf.get_variable('moving_variance', params_shape, tf.float32,
                                                  initializer=tf.constant_initializer(1.0, tf.float32),
                                                  trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                                       initializer=tf.constant_initializer(0.0, tf.float32),
                                       trainable=False)
                variance = tf.get_variable('moving_variance', params_shape, tf.float32,
                                           initializer=tf.constant_initializer(1.0, tf.float32),
                                           trainable=False)
                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)

            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    def _bottleneck_residual(self, x, in_filter, out_filter, stride):
        with tf.variable_scope('residual_bn_relu'):
            original_x = x
            x = self._batch_norm('bn1', x)
            x = self._relu(x)

        with tf.variable_scope('bottleneck_sub1'):
            # 第一个stride不确定是1还是2
            x = self._conv('conv1', x, filter_size=1, in_filters=in_filter, out_filters=out_filter//4, strides=stride)

        with tf.variable_scope('bottleneck_sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x)
            x = self._conv('conv2', x, filter_size=3, in_filters=out_filter//4, out_filters=out_filter//4, strides=[1, 1, 1, 1])

        with tf.variable_scope('bottleneck_sub3'):
            x = self._batch_norm('bn3', x)
            x = self._relu(x)
            x = self._conv('conv3', x, filter_size=1, in_filters=out_filter//4, out_filters=out_filter, strides=[1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                original_x = self._conv('project', original_x, filter_size=1, in_filters=in_filter, out_filters=out_filter, strides=stride)
                #original_x = tf.pad(original_x, [[0, 0], [0, 0], [0, 0]])
                # 注意图像高度和宽度不同小心
            x += original_x

        tf.logging.info('image after unit %s', x.get_shape())
        return x

    def _build_model(self):
        with tf.variable_scope('init'):
            x = self.images
            x =self._conv('init_conv', x, filter_size=7, in_filters=3, out_filters=64, strides=self._stride_arr(2))

        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool')

        with tf.variable_scope('unit_2_1'):
            x = self._bottleneck_residual(x, in_filter=64, out_filter=256, stride=[1, 1, 1, 1])
        for i in range(1, self.hps.conv2_nums):
            with tf.variable_scope('unit_2_%d' %(i+1)):
                x = self._bottleneck_residual(x, in_filter=256, out_filter=256, stride=[1, 1, 1, 1])

        with tf.variable_scope('unit_3_1'):
            x = self._bottleneck_residual(x, in_filter=256, out_filter=512, stride=[1, 2, 2, 1])
        for i in range(1, self.hps.conv3_nums):
            with tf.variable_scope('unit_3_%d' %(i+1)):
                x = self._bottleneck_residual(x, in_filter=512, out_filter=512, stride=[1, 1, 1, 1])

        with tf.variable_scope('unit_4_1'):
            x = self._bottleneck_residual(x, in_filter=512, out_filter=1024, stride=[1, 2, 2, 1])
        for i in range(1, self.hps.conv4_nums):
            with tf.variable_scope('unit_4_%d' %(i+1)):
                x = self._bottleneck_residual(x, in_filter=1024, out_filter=1024, stride=[1, 1, 1, 1])

        with tf.variable_scope('unit_5_1'):
            x = self._bottleneck_residual(x, in_filter=1024, out_filter=2048, stride=[1, 2, 2, 1])
        for i in range(1, self.hps.conv5_nums):
            with tf.variable_scope('unit_5_%d' %(i+1)):
                x = self._bottleneck_residual(x, in_filter=2048, out_filter=2048, stride=[1, 1, 1, 1])

        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x)
            x = self._global_avg_pool(x)

        with tf.variable_scope('logit'):
            logits = self._fully_connected(x, self.hps.num_classes)
            self.logits = logits
            self.predictions = tf.nn.softmax(logits)

        with tf.variable_scope('costs'):
            xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
            self.cost = tf.reduce_mean(xent, name='xent')
            self.cost += self._decay()

            tf.summary.scalar('cost', self.cost)


    def _build_train_op(self):
        self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        tf.summary.scalar('learning_rate', self.lrn_rate)

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.cost, trainable_variables)

        optimizer = tf.train.AdamOptimizer(self.lrn_rate)

        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=self.global_step, name='train_step')

        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)
