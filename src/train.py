"""Codes for train and evaluate model"""

import os, sys
# add parent directory into sys.path
#code_path = os.path.abspath(__file__)
#parent_path = os.path.dirname(code_path)
# TODO
parent_path = os.path.abspath('../')
sys.path.insert(0, parent_path)

import time
import tensorflow as tf
import numpy as np
import pandas as pd
import coloredlogs
from datetime import datetime
from collections import namedtuple

# auxiliary modules
try:
    import auxiliary.timer as timer
    import auxiliary.statistics as statistics
    #import auxiliary.process_bar as process_bar
except:
    print('import from parent_folder error')
    sys.exit(-1)

import data_utils
import model
import memory


# hyperparameters
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', 'train',
                       """train or evaluate""")
tf.flags.DEFINE_string('model', 'ResNet',
                       """AlexNet or Vgg19 or ResNet""")
tf.flags.DEFINE_bool('remove', False,
                     """remove logs and parameters""")
tf.flags.DEFINE_integer('epoch', 200,
                        """Counts to run all the images""")
tf.flags.DEFINE_string('data', 'ISIC2018',
                       """ISIC2018 or cifar10""")
tf.flags.DEFINE_integer('start_k', 4,
                        """start k from""")
tf.flags.DEFINE_integer('valid_frequency', 9,
                        """valid_frequency valid at % valid_frequency, at least from 1!!Less than epoch""")
tf.flags.DEFINE_integer('feature_size', 64,
                        """feature size""")
tf.flags.DEFINE_string('with_memory', False,
                       """integrate memory to train model""")


# where to put log and parameters
tf.flags.DEFINE_string('save_prefix', '../save_{}_{}_{}',
                        """save model, logs and extra files' directory prefix""")
tf.flags.DEFINE_string('logdir', 'logs/',
                       """Directory where to write graph logs """)
tf.flags.DEFINE_string('parameters', 'parameters/',
                       """Directory where to write event logs """
                       """and checkpoint.""")
tf.flags.DEFINE_string('checkpoint_dir', 'parameters/model_train_{}_{}_{}',
                       """checkpoint_dir""")
tf.flags.DEFINE_string('evaluation_detail', 'evaluation_detail.csv',
                        """save evaluation detail""")

# constants
tf.flags.DEFINE_bool('log_device_placement', False,
                     """Whether to log device placement.""")
tf.flags.DEFINE_string('device_cpu', '/cpu:0',
                        """Using cpu to compute""")
tf.flags.DEFINE_string('device_gpu', '/device:GPU:',
                        """Using GPU to compute""")
# use particular GPU
tf.flags.DEFINE_string('CUDA_VISIBLE_DEVICE', '0',
                                """CUDA_VISBLE_DEVICE""")

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.CUDA_VISIBLE_DEVICE


HParams = namedtuple('HParams',
                     'model,'
                     'feature_size, batch_size, num_classes, '
                     'num_residual_units, use_bottleneck, '
                     'weight_decay_rate, relu_leakiness, optimizer, '
                     'weight_sample, use_weight_sample, using_pretrained')

def train(hps, val_index):
    """Train model for a number of steps"""

    # select dataset
    with tf.device(FLAGS.device_cpu):
        if FLAGS.data == 'ISIC2018':
            data = data_utils.ISIC2018_data()
        else:
            tf.logging.info('Give dataset name')
            sys.exit(-1)

        # get data information: width, height from data_utils [224, 224, 3]
        width, height, channel = data.get_shape()
        # get data information: owidth, oheight from data_utils [400, 300, 3]
        owidth, oheight, ochannel = data.get_origin_shape()

        save_prefix = FLAGS.save_prefix.format(FLAGS.batch_size, oheight, owidth)
        parameters = os.path.join(save_prefix, FLAGS.parameters)
        logdir = os.path.join(save_prefix, FLAGS.logdir)

        psuffix = 'model_' + FLAGS.mode + '_' + FLAGS.model + '_' + str(FLAGS.with_memory) + '_' + str(val_index)
        lsuffix = 'log_' + FLAGS.mode + '_' + FLAGS.model + '_' + str(FLAGS.with_memory) + '_' + str(val_index)
        train_model = os.path.join(parameters, psuffix)
        train_graph = os.path.join(logdir, lsuffix)

        tf.logging.info('train model %s' % train_model)
        tf.logging.info('train graph %s' % train_graph)

        if FLAGS.remove:
            if tf.gfile.Exists(train_model):
                tf.gfile.DeleteRecursively(train_model)
                tf.gfile.MakeDirs(train_model)
            if tf.gfile.Exists(train_graph):
                tf.gfile.DeleteRecursively(train_graph)
                tf.gfile.MakeDirs(train_graph)
            sepoch = 0
        else:
            ckpt = tf.train.get_checkpoint_state(train_model)
            if ckpt and ckpt.model_checkpoint_path:
                sepoch = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[1]) + 1
                model_path = ckpt.model_checkpoint_path
                tf.logging.info(model_path)
                tf.logging.info('sepoch is %d' % sepoch)
            else:
                sepoch = 0
                tf.logging.info('Fail to restore, start from %d' % sepoch)

        if FLAGS.with_memory:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir.format(FLAGS.batch_size, data.nHei, data.nWid, val_index))
            if ckpt and ckpt.model_checkpoint_path:
                all_model_checkpoint_paths = ckpt.all_model_checkpoint_paths
                model_path_template = all_model_checkpoint_paths[0].split('-')[0]
                model_path = model_path_template + '-99'
                sepoch = 0
                tf.logging.info(model_path)
                tf.logging.info('sepoch is %d' % sepoch)
            else:
                tf.logging.info('Fail to restore for memory')
                sys.exit(-1)


    gpu = FLAGS.device_gpu + str(0)
    with tf.device(gpu):
        with tf.Graph().as_default() as g:
            # validation groups
            data.set_valid_index(val_index)
            # define variables
            xs = tf.placeholder(tf.float32, [None, height, width, channel], name='xs')
            ys = tf.placeholder(tf.float32, [None, hps.num_classes], name='ys')
            trainable = tf.placeholder(tf.bool, name='trainable')
            learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
            #train_mode = tf.placeholder(tf.bool)

            # using model
            selected_model =  model.get_model(hps, xs, ys, learning_rate, trainable)
            selected_model.build_graph()

            # TODO Use memory
            if FLAGS.with_memory:
                memory_layer = memory.Memory(hps, key_dim=FLAGS.feature_size)
                train_op = memory_layer.query_and_make_update(model.feature_map, ys)
                summaries = tf.summary.merge_all()
                predictions = memory_layer.query(model.feature_map)
                get_restored_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                        scope='extract_feature_map')
                saver = tf.train.Saver(get_restored_variables ,max_to_keep=FLAGS.epoch)
                #init_op = tf.variables_initializer([memory_layer.mem_keys, memory_layer.mem_vals, memory_layer.mem_age, memory_layer.query_proj, memory_layer.threshold], name='init_op')
            else:
                # default to use this branch now,
                # model provided at least train_op, summaries and predictions
                train_op = selected_model.train_op
                summaries = selected_model.summaries
                predictions = selected_model.predictions
                saver = tf.train.Saver(max_to_keep=26)
                init_op = tf.global_variables_initializer()

            config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
            config.gpu_options.allow_growth = True

            summary_writer = tf.summary.FileWriter(train_graph, g)

            step = 0
            with tf.Session(config=config) as sess:
                if FLAGS.with_memory == False and (FLAGS.remove == True or sepoch ==  0):
                    sess.run(init)
                else:
                    sess.run(init)
                    saver.restore(sess, model_path)

                # training for (FALGS.epoch-sepoch)
                for i in range(sepoch, FLAGS.epoch):
                    # init iterator for dataset
                    sess.run(data.extra_init)
                    tf.logging.info("%s: Epoch %d, val_index is %d" % (datetime.now(), i, val_index))
                    # set learning rate
                    if i < 40:
                        learning_rate_ = 0.1
                    elif i < 90:
                        learning_rate_ = 0.01
                    elif i < 150:
                        learning_rate_ = 0.001
                    else:
                        learning_rate_ = 0.0001

                    batch_x_iter, batch_y_iter, path_iter = data.get_inputs()
                    try:
                        j = 0
                        while True:
                            batch_x, batch_y, path = sess.run([batch_x_iter, batch_y_iter, path_iter])
                            _, summary_op_ = sess.run(
                                    [train_op, summaries],
                                    feed_dict={ xs:batch_x,
                                                ys:batch_y,
                                                learning_rate: learning_rate_,
                                                trainable: True
                                              })
                            summary_writer.add_summary(summary_op_, step)
                            step += 1
                            j += 1
                    except tf.errors.OutOfRangeError:
                        tf.logging.info("%s: Epoch %d complete after iter %d" % (datetime.now(), i, j))

                    # Val
                    if i % 5 == FLAGS.valid_frequency:
                        batch_x_iter, batch_y_iter, path_iter = data.get_train()
                        try:
                            while True:
                                batch_x, batch_y, path = sess.run([batch_x_iter, batch_y_iter, path_iter])
                                predictions_ = sess.run(
                                            predictions,
                                            feed_dict={
                                                xs:batch_x,
                                                ys:batch_y,
                                                trainable: False})
                                statistics_.add_labels_predictions(batch_y, predictions_, path)
                        except tf.errors.OutOfRangeError:
                            tf.logging.info("%s: Epoch %d training complete " % (datetime.now(), i))
                            statistics_.get_print_precision()
                            acc_normal = statistics_.gey_acc_normal()
                            tf.logging.info("%s: Epoch %d complete acc_normal is %f" % (datetime.now(), i, acc_normal))
                            acc_imbalanced = statistics_.get_acc_imbalanced()
                            tf.logging.info("%s: Epoch %d complete acc_imbalanced is %f" % (datetime.now(), i, acc_imbalanced))

                        saver.save(sess, os.path.join(train_model, 'model'), global_step=i+1)



def evaluate(hps):
    """evaluate"""
    sec = 150 * 6
    tf.logging.info('Evaluate every %ds' % sec)

    # select dataset
    with tf.device(FLAGS.device_cpu):
        if FLAGS.data == 'ISIC2018':
            data = data_utils.ISIC2018_data()
        else:
            tf.logging.info('Give dataset name')
            sys.exit(-1)

        # get data information: width, height from data_utils [224, 224, 3]
        width, height, channel = data.get_shape()
        # get data information: owidth, oheight from data_utils [400, 300, 3]
        owidth, oheight, ochannel = data.get_origin_shape()

        save_prefix = FLAGS.save_prefix.format(FLAGS.batch_size, oheight, owidth)
        checkpoint_dir = FLAGS.checkpoint_dir.format(FLAGS.batch_size, oheight, owidth)
        checkpoint_dir = os.path.join(save_prefix, checkpoint_dir)

    gpu = FLAGS.device_gpu + str(0)
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    while True:
        with tf.device(gpu):
            with tf.Session(config=config) as sess:
                # get graph
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    epoch = int(ckpt.model_checkpoint_path.split('-')[1])
                    model_path = ckpt.model_checkpoint_path
                    graph_path = os.path.join(FLAGS.checkpoint_dir, model_path+'.meta')
                    tf.logging.info('Restore graph from %s, Epoch is %d' % (graph_path, epoch))
                else:
                    tf.logging.info('Fail to restore, Sleep for %d' % sec)
                    time.sleep(sec)
                    continue

                statistics_ = statistics.statistics(hp, epoch,  mode='evaluate')

                # load graph from checkpoint and load variable
                saver = tf.train.import_meta_graph(graph_path)
                saver.restore(sess, model_path)

                # get desirable variables
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)[0]
                predictions = tf.get_collection('predictions')[0]
                train_op = tf.get_collction('train_op')[0]
                trainable = tf.get_collection('trainable')[0]
                xs = tf.get_collection('xs')[0]
                ys = tf.get_collection('ys')[0]
                #summaries = tf.get_tensor_by_name(summaries+':0')
                #predictions = tf.get_tensor_by_name('predictions:0')
                #train_op = tf.get_tensor_by_name('train_op:0')

                batch_x_iter, batch_y_iter, path_iter = data.get_valid()
                try:
                    while True:
                        batch_x, batch_y, path = sess.run([batch_x_iter, batch_y_iter, path_iter])
                        predictions_ = sess.run(
                            predictions,
                            feed_dict={
                                xs:batches,
                                ys:batch_y,
                                trainable: False})
                        statistics_.add_labels_predictions(batch_y, predictions_, path)
                except tf.errors.OutOfRangeError:
                    tf.logging.info("%s: Epoch %d evaluation complete " % (datetime.now(), i))
                    statistics_.get_print_precision()
                    acc_normal = statistics_.gey_acc_normal()
                    tf.logging.info("%s: Epoch %d complete acc_normal is %f" % (datetime.now(), i, acc_normal))
                    acc_imbalanced = statistics_.get_acc_imbalanced()
                    tf.logging.info("%s: Epoch %d complete acc_imbalanced is %f" % (datetime.now(), i, acc_imbalanced))

                evaluation_detail = os.path.join(save_prefix, FLAGS.evaluation_detail)
                statistics_.save_triples(evaluation_detail)

                time.sleep(sec)

def main(argv=None):
    """Load data and run train"""

    hps = HParams(
             model=FLAGS.model,
             feature_size=FLAGS.feature_size,
             batch_size=FLAGS.batch_size,
             num_classes=7,
             num_residual_units=[1,3,4,6,3],
             use_bottleneck=False,
             weight_decay_rate=0.0002,
             relu_leakiness=0.1,
             optimizer='mom',
             weight_sample=weight_sample_,
             use_weight_sample=True,
             using_pretrained=False)

    if FLAGS.mode == 'train':
        k_fold = FLAGS.k_fold
        timer_ = timer.timer()
        if FLAGS.start_k == -1:
            start_k = 0
        else:
            start_k = FLAGS.start_k
        for i in range(start_k, k_fold):
            tf.logging.info('%s K-Cross-Validation: %d' % (datetime.now(), i))
            train(hps, i)
        timer_.get_duration()
    elif FLAGS.mode == 'evaluate':
        timer_ = timer.timer()
        evaluate(hps)
        timer_.get_duration()


if __name__ == '__main__':
    coloredlogs.install(level='INFO')
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
