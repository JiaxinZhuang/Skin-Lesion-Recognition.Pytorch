"""Implementation of resnet
    python == 3.4
    tensorflow version == 1.4.0

"""

from collections import namedtuple

import tensorflow as tf
import numpy as np

from tensorflow.python.training import moving_averages


#HParams = namedtuple('HParams',
#                     'feature_size, batch_size, num_classes, '
#                     'num_residual_units, use_bottleneck, '
#                     'weight_decay_rate, relu_leakiness, optimizer, '
#                     'weight_sample, use_weight_sample, using_pretrained')


class ResNet():
    def __init__(self, hps, images, labels, learning_rate, trainable):
        """ResNet constructor

        Args:
            hps:
            images: [batch_size, image_size, image_size, 3]
            labels: [batch_size, num_classes]
            mode : 'train' or 'eval'
        """
        self.hps = hps
        self.image_size=224
        self._images = self._pre_process_images(images)
        self.labels = labels
        self.mode = 'train'
        self.lrn_rate = learning_rate
        self.trainable = trainable
        # TODO
        self.weights_sample = tf.constant(self.hps.weight_sample)

        self._extra_train_ops = []

        #if self.hps.using_pretrained:
        #    self.data_dict = np.load(resnet_npz_path)
        #else:
        #    self.data_dict = None

    def _pre_process_images(self, x):
        images = tf.image.resize_image_with_crop_or_pad(x, self.image_size+4, self.image_size+4)
        images = tf.map_fn(lambda img: tf.random_crop(img, [self.image_size, self.image_size, 3]), images)
        images = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), images)
        return images

    def build_graph(self):
        self.global_step = tf.train.get_or_create_global_step()
        self._build_model()
        if self.mode == 'train':
            self._build_train_op()
        self.summaries = tf.summary.merge_all()

    def _build_model(self):
        """Build the core model within the graph"""
        with tf.variable_scope('extract_feature_map'):

            strides = [2, 1, 2, 2, 2]
            activate_before_residual = [True, False, False, False]
            # TODO
            if self.hps.use_bottleneck:
                res_func = self._bottleneck_residual
                filters = [16, 64, 128, 256]
            else:
                res_func = self._residual
                filters = [64, 64, 128, 256, 512]

            with tf.variable_scope('unit_1'):
                x = self._images
                x = self._conv('unit_1', x, 7, 3, filters[0], self._stride_arr(strides[0]))

            with tf.variable_scope('unit_2_0'):
                x = tf.nn.max_pool(x, [1,3,3,1], [1,2,2,1], padding='SAME', name='max_pool')
                x = res_func(x, filters[0], filters[1], self._stride_arr(strides[1]),
                        activate_before_residual[0])
            # num_residual_units is a list [0,2,3,5,2]
            for i in range(1, self.hps.num_residual_units[1]):
                with tf.variable_scope('unit_1_%d' % i):
                    x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

            with tf.variable_scope('unit_3_0'):
                x = res_func(x, filters[1], filters[2], self._stride_arr(strides[2]),
                        activate_before_residual[1])
            for i in range(1, self.hps.num_residual_units[2]):
                with tf.variable_scope('unit_3_%d' % i):
                    x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

            with tf.variable_scope('unit_4_0'):
                x = res_func(x, filters[2], filters[3], self._stride_arr(strides[3]),
                        activate_before_residual[2])
            for i in range(1, self.hps.num_residual_units[3]):
                with tf.variable_scope('unit_4_%d' %i):
                    x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

            with tf.variable_scope('unit_5_0'):
                x = res_func(x, filters[3], filters[4], self._stride_arr(strides[4]),
                        activate_before_residual[2])
            for i in range(1, self.hps.num_residual_units[4]):
                with tf.variable_scope('unit_5_%d' %i):
                    x = res_func(x, filters[4], filters[4], self._stride_arr(1), False)

            with tf.variable_scope('unit_last'):
                x = self._batch_norm('final_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
                self.feature_map = self._global_avg_pool(x)

        with tf.variable_scope('logit'):
            logits = self._fully_connected(self.feature_map, self.hps.num_classes)
            self.predictions  = tf.nn.softmax(logits, name='predictions')

        with tf.variable_scope('costs'):
            if self.hps.use_weight_sample:
		# Median-class weight
                #xent = self._median_weight_class_loss(self.labels, logits)
		# Focal loss class weight
                xent = self._focal_loss(self.labels, logits)
            else:
                xent = tf.nn.softmax_cross_entropy_with_logits(
                        logits=logits, labels=self.labels)

            self.cost = tf.reduce_mean(xent, name='xent')
            self.cost += self._decay()

            tf.summary.scalar('cost', self.cost)


    def _median_weight_class_loss(self, labels, logits):
        epsilon = tf.constant(value=1e-10)
        logits = logits + epsilon
        softmax = tf.nn.softmax(logits)

        xent = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), self.hps.weight_sample), axis=1, keep_dims=True)
        return xent


    def _focal_loss(self, labels, logits, gamma=2.0, alpha=4.0):
        epsilon = tf.constant(value=1e-9)
        softmax = tf.nn.softmax(logits)
        model_out = tf.add(softmax, epsilon)
        ce = tf.multiply(labels, -tf.log(model_out))
        weight = tf.multiply(labels, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_sum(fl, axis=1, keep_dims=True)

        return reduced_fl


    def _bottleneck_residual(self, x, in_filter, out_filter, stride, activate_before_residual=False):
        """Bottleneck residual unit with 3 sub layers"""
        if activate_before_residual == True:
            with tf.variable_scope('common_bn_relu'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_bn_relu'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.leakiness)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 1, in_filter, out_filter/4, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv2', x, 3, out_filter/4, out_filter/4. [1,1,1,1])

        with tf.variable_scope('sub3'):
            x = self._batch_norm('bn3', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv3', x, 1, out_filter/4, out_filter, [1,1,1,1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
            x += orig_x

        tf.logging.info('image after unit %s', (tf.shape(x),))
        return x

    def _residual(self, x, in_filter, out_filter, stride,
            activate_before_residual=False):
        """Residual unit with 2 sub layers"""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1,1,1,1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'SAME')
                orig_x = tf.pad(
                        orig_x, [[0,0], [0,0], [0,0],
                            [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
            x += orig_x

        #tf.logging.info('image after unit %s'% (tf.shape(x),))
        return x

    def _build_train_op(self):
        """Build training specific ops for the graph"""
        #self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        tf.summary.scalar('learning_rate', self.lrn_rate)

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.cost, trainable_variables)

        if self.hps.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)

        apply_op = optimizer.apply_gradients(
                zip(grads, trainable_variables),
                global_step=self.global_step, name='train_op')

        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops, name='train_op')


    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d"""
        return [1, stride, stride, 1]

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution"""
        with tf.variable_scope(name):
            kernel = self._get_conv_var(filter_size, in_filters, out_filters, name)
        return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _get_conv_var(self, filter_size, in_filters, out_filters, name):
        n = filter_size * filter_size * out_filters
        initial_value = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0/n)))
        #kernel = self._get_var(initial_value, name, "DW")
        return initial_value

    def _batch_norm(self, name, x):
        """Batch Normalization"""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                    'beta', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                    'gamma', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0,1,2], name='moments')

                moving_mean = tf.get_variable(
                        'moving_mean', params_shape, tf.float32,
                        initializer=tf.constant_initializer(0.0, tf.float32),
                        trainable=False)
                moving_variance = tf.get_variable(
                        'moving_variance', params_shape, tf.float32,
                        initializer=tf.constant_initializer(1.0, tf.float32),
                        trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                        'moving_mean', params_shape, tf.float32,
                        initializer=tf.constant_initializer(0.0, tf.float32),
                        trainable=False)
                variance = tf.get_variable(
                        'moving_variance', params_shape, tf.float32,
                        initializer=tf.constant_initializer(1.0, tf.float32),
                        trainable=False)
                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)

            y = tf.nn.batch_normalization(
                    x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support"""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(self, x, out_dim):
        """Fully Connected layer for final output"""
        # [batch_size, 64]
        #x = tf.reshape(x, [self.hps.batch_size, -1])
        #print(tf.shape(x))
        #print(x.get_shape())
        w = tf.get_variable(
                'DW', [x.get_shape()[1], out_dim],
                initializer=tf.initializers.variance_scaling(scale=1.0))
                #initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1,2])

    def _decay(self):
        """L2 weight decay loss"""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    def _get_var(self, initial_value, name, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var
