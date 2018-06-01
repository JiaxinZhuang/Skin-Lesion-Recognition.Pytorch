import tensorflow as tf

import numpy as np
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
    """
    A trainable version VGG19.
    """

    def __init__(self, hps, xs, ys, learning_rate, vgg19_npy_path=None, trainable=True, dropout=0.5):
        self.hps = hps
        self._images = xs
        self.labels = ys
        self.lrn_rate = learning_rate
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout

        self.image_size = 224

        self._extra_train_ops = []

    def build_graph(self, train_mode=None):
        self.train_mode = train_mode
        self.global_step = tf.train.get_or_create_global_step()
        self._build_model(train_mode=train_mode)
        self._build_train_op()
        self.summaries = tf.summary.merge_all()

    def _build_model(self, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0,255]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        #rgb_scaled = rgb * 255.0

        bgr = self._images
        bgr = self._pre_process_images(bgr)

        # Convert RGB to BGR
        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=bgr)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]


        with tf.variable_scope('extract_feature_map'):
            self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
            self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

            self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
            self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')

            self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
            self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
            self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
            self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
            self.pool3 = self.max_pool(self.conv3_4, 'pool3')

            self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
            self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
            self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
            self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
            self.pool4 = self.max_pool(self.conv4_4, 'pool4')

            self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
            self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
            self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
            self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
            self.pool5 = self.max_pool(self.conv5_4, 'pool5')

            self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
            self.relu6 = tf.nn.relu(self.fc6)
            if train_mode is not None:
                self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
            elif self.trainable:
                self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

            self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
            self.relu7 = tf.nn.relu(self.fc7)
            if train_mode is not None:
                self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
            elif self.trainable:
                self.relu7 = tf.nn.dropout(self.relu7, self.dropout)
            self.fc8 = self.fc_layer(self.relu7, 4096, 1000, "fc8")

            with tf.variable_scope('logits'):
                self.logits = self._fully_connected(self.fc8, self.hps.num_classes, "logits")
                self.predictions = tf.nn.softmax(self.logits, name="prob")

            with tf.variable_scope('costs'):
                xent = tf.nn.softmax_cross_entropy_with_logits(
                        logits=self.logits, labels=self.labels)

                self.costs = tf.reduce_mean(xent, name='xent')
                self.costs += self._decay()

                tf.summary.scalar('cost', self.costs)


            self.data_dict = None

    def _fully_connected(self, x, out_dim, name):
        with tf.variable_scope(name):
            w = tf.get_variable(
                    'DW', [x.get_shape()[1], out_dim],
                    initializer=tf.initializers.variance_scaling(scale=1.0))
                    #initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
            b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
            return tf.nn.xw_plus_b(x, w, b)


    def _build_train_op(self):
        tf.summary.scalar('learning_rate', self.lrn_rate)

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.costs, trainable_variables)

        if self.hps.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)

        apply_op = optimizer.apply_gradients(
                zip(grads, trainable_variables),
                global_step=self.global_step, name='train_op')

        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)


    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count

    def _decay(self):
        """L2 weight decay loss"""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'_filters') > 0 or var.op.name.find(r'_weights') > 0 or \
                    var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))


    def _pre_process_images(self, x):
        if self.train_mode == True:
            images = tf.image.resize_image_with_crop_or_pad(x, self.image_size+4, self.image_size+4)
            images = tf.map_fn(lambda img: tf.random_crop(img, [self.image_size, self.image_size, 3]), images)
            images = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), images)
        else:
            images = tf.image.resize_image_with_crop_or_pad(x, self.image_size, self.image_size)
        return images
