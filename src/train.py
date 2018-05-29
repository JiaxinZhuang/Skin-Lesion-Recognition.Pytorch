"""Codes for train and evaluate model"""

from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
import coloredlogs

# auxiliary modules
import data_utils
#import model
import timer
#import process_bar

import memory
import resNet

FLAGS = tf.flags.FLAGS

# hyperparameters
tf.flags.DEFINE_string('mode', 'train',
                       """train or test""")
tf.flags.DEFINE_string('model', 'ResNet',
                       """Cnn or Vgg or ResNet""")
tf.flags.DEFINE_string('with_memory', False,
                       """integrate memory to train model""")
tf.flags.DEFINE_integer('epoch', 150,
                        """Counts to run all the images""")
tf.flags.DEFINE_bool('remove', False,
                     """remove logs and parameters""")
tf.flags.DEFINE_string('data', 'ISIC2018',
                       """ISIC2018 or cifar10""")
tf.flags.DEFINE_integer('feature_size', 64,
                        """feature size""")
tf.flags.DEFINE_integer('start_k', -1,
                        """start k from""")
tf.flags.DEFINE_integer('valid_frequency', 9,
                        """valid_frequency valid at % valid_frequency, at least from 1!!Less than epoch""")

# where to put log and parameters
tf.flags.DEFINE_string('logdir', '../save_{}_{}_{}_CW/logs/',
                       """Directory where to write graph logs """)
tf.flags.DEFINE_string('parameters', '../save_{}_{}_{}_CW/parameters/',
                       """Directory where to write event logs """
                       """and checkpoint.""")
tf.flags.DEFINE_string('checkpoint_dir', '../save_{}_{}_{}_CW/parameters/model_train_ResNet_False_{}',
                       """checkpoint_dir""")

# constants
tf.flags.DEFINE_bool('log_device_placement', False,
                     """Whether to log device placement.""")
tf.flags.DEFINE_string('device_cpu', '/cpu:0',
                        """Using cpu to compute""")
tf.flags.DEFINE_string('device_gpu', '/device:GPU:',
                        """Using GPU to compute""")
tf.flags.DEFINE_string('CUDA_VISIBLE_DEVICE', '0',
                                """CUDA_VISBLE_DEVICE""")

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.CUDA_VISIBLE_DEVICE



def train(hps, val_index):
    """Train model for a number of steps"""

    # select dataset
    with tf.device(FLAGS.device_cpu):
        if FLAGS.data == 'ISIC2018':
            data = data_utils.ISIC2018_data()
        else:
            tf.logging.info('Give dataset name')
            sys.exit(-1)
        # get data information, width, height from data_utils
        width, height, channel = data.get_shape()
        data.set_valid_index(val_index)

    parameters = FLAGS.parameters.format(FLAGS.batch_size, data.nHei, data.nWid)
    logdir = FLAGS.logdir.format(FLAGS.batch_size, data.nHei, data.nWid)
    #tf.logging.info('Epoch is %d' % FLAGS.epoch)
    psuffix = 'model_' + FLAGS.mode + '_' + FLAGS.model + '_' + str(FLAGS.with_memory) + '_' + str(val_index)
    lsuffix = 'log_' + FLAGS.mode + '_' + FLAGS.model + '_' + str(FLAGS.with_memory) + '_' + str(val_index)
    train_model = os.path.join(parameters, psuffix)
    train_graph = os.path.join(logdir, lsuffix)

    tf.logging.info('train model %s' % train_model)
    tf.logging.info('train graph %s' % train_graph)

    tf.logging.info('Remove save parameters and logs before: %r' % FLAGS.remove)
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
            #global_step = tf.train.get_or_create_global_step()

            # define variables
            xs = tf.placeholder(tf.float32, [None, width, height, channel])
            ys = tf.placeholder(tf.float32, [None, hps.num_classes])
            learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
            #val = tf.placeholder(tf.float32)


           #image = tf.image.resize_image_with_crop_or_pad(
           #     image, image_size+4, image_size+4)
           #image = tf.random_crop(image, [image_size, image_size, 3])
           #image = tf.image.random_flip_left_right(image)

            resnet_model = resNet.ResNet(hps, xs, ys, 'train', learning_rate)
            resnet_model.build_graph()
            # Use memory
            if FLAGS.with_memory:
                memory_layer = memory.Memory(hps, key_dim=FLAGS.feature_size)
                train_op = memory_layer.query_and_make_update(resnet_model.feature_map, ys)
                summaries = tf.summary.merge_all()
                predictions = memory_layer.query(resnet_model.feature_map)
            else:
                train_op = resnet_model.train_op
                summaries = resnet_model.summaries
                predictions = resnet_model.predictions
            # define train process
            #feature_vec = model.inference(xs, FLAGS.model)
            #    logits = model.memory(feature_vec)
            #else:
            #    logits = model.softmax_linear_layer(feature_vec)
            #loss = model.loss(logits, ys, val)
            #train_op = model.train(loss, global_step)

            init = tf.global_variables_initializer()

            config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
            config.gpu_options.allow_growth = True

            if FLAGS.with_memory:
                get_restored_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                        scope='extract_feature_map')
                saver = tf.train.Saver(get_restored_variables ,max_to_keep=FLAGS.epoch)
            else:
                saver = tf.train.Saver(max_to_keep=FLAGS.epoch)

            summary_writer = tf.summary.FileWriter(train_graph, g)

            #init_op = tf.variables_initializer([memory_layer.mem_keys, memory_layer.mem_vals, memory_layer.mem_age, memory_layer.query_proj, memory_layer.threshold], name='init_op')
            step = 0
            with tf.Session(config=config) as sess:
                if FLAGS.with_memory == False and (FLAGS.remove == True or sepoch ==  0):
                    sess.run(init)
                else:
                    sess.run(init)
                    saver.restore(sess, model_path)

                for i in range(sepoch, FLAGS.epoch):
                    tf.logging.info("%s: Epoch %d, val_index is %d" % (datetime.now(), i, val_index))
                    if i < 60:
                        learning_rate_ = 0.1
                    elif i < 100:
                        learning_rate_ = 0.01
                    elif i < 130:
                        learning_rate_ = 0.001
                    else:
                        learning_rate_ = 0.0001

                    # train
                    for j in range(data.k_fold):
                        if j != val_index:
                            for batch_x, batch_y in data.get_groups(j):
                                _, summary_op_ = sess.run(
                                        [train_op, summaries],
                                        feed_dict={ xs:batch_x,
                                                    ys:batch_y,
                                                    learning_rate: learning_rate_
                                                    })
                                summary_writer.add_summary(summary_op_, step)
                                step += 1

                    if i % 10 == FLAGS.valid_frequency:
                        counter = [ 0 for _ in range(hps.num_classes)]
                        class_nums = [ 0 for _ in range(hps.num_classes)]
                        pre_class_nums = [ 0 for _ in range(hps.num_classes)]
                        dicts = []
                        all_imgs = 0.0
                        all_cor = 0.0
                        for j in range(data.k_fold):
                            if j == val_index:
                                continue
                            for batch_x, batch_y in data.get_groups(j):
                                predictions_ = sess.run(
                                            predictions,
                                            feed_dict={
                                                xs:batch_x,
                                                ys:batch_y})
                                assert predictions_.shape == np.array(batch_y).shape
                                batch_y_ = np.argmax(batch_y, axis=1)
                                predictions_ = np.argmax(predictions_, axis=1)
                                all_imgs += len(batch_y_)
                                for y, ypred in zip(batch_y_, predictions_):
                                    # recall
                                    class_nums[y] += 1
                                    # precision
                                    pre_class_nums[ypred] += 1
                                    dicts.append((y, ypred))
                                    if y == ypred:
                                        counter[y] += 1
                                        all_cor += 1
                        output_filename = '../save_{}_{}_{}_CW/tra/val_index_{}_y_ypre_{}'.format(FLAGS.batch_size, data.nHei, data.nWid, val_index, i)
                        with open(output_filename, 'wb') as fo:
                            np.save(fo, dicts)

                        recalls = []
                        precisions = []
                        for k in range(hps.num_classes):
                            if class_nums[k] == 0:
                                recall = 0
                            else:
                                recall = counter[k]/class_nums[k]
                            recalls.append(recall)
                            if pre_class_nums[k] == 0:
                                precision = 0
                            else:
                                precision = counter[k]/pre_class_nums[k]
                            precisions.append(precision)

                            #tf.logging.info('%s: Recall of class %d is: %.4f' % (datetime.now(), k, recall))
                            tf.logging.info('%s: Training Precision of class %d is: %.4f' % (datetime.now(), k, precision))
                        accuracy = (all_cor*1.0)/all_imgs
                        tf.logging.info('%s: Training Acc of all classes is %.4f' % (datetime.now(), accuracy))

                        output_filename = '../save_{}_{}_{}_CW/tra/val_index_{}_recall_precision_{}.csv'.format(FLAGS.batch_size, data.nHei, data.nWid, val_index, i)
                        col_names = ['recall', 'precision', 'accuracy']
                        acc = [accuracy] * len(precisions)
                        output = {'recall': recalls, 'precision': precisions, 'accuracy': acc}
                        df = pd.DataFrame(output, columns=col_names)
                        with open(output_filename, 'w', newline="") as fo:
                            df.to_csv(fo)


                    assert FLAGS.valid_frequency >= 0
                    if i % 10 == FLAGS.valid_frequency:
                        # val all Accuracy, precision ...
                        counter = [ 0 for _ in range(hps.num_classes)]
                        class_nums = [ 0 for _ in range(hps.num_classes)]
                        pre_class_nums = [ 0 for _ in range(hps.num_classes)]
                        dicts = []
                        all_imgs = 0.0
                        all_cor = 0.0
                        for batch_x, batch_y in data.get_groups(val_index):
                            predictions_ = sess.run(
                                        predictions,
                                        feed_dict={
                                            xs:batch_x,
                                            ys:batch_y})
                            assert predictions_.shape == np.array(batch_y).shape
                            batch_y_ = np.argmax(batch_y, axis=1)
                            predictions_ = np.argmax(predictions_, axis=1)
                            all_imgs += len(batch_y_)
                            for y, ypred in zip(batch_y_, predictions_):
                                # recall
                                class_nums[y] += 1
                                # precision
                                pre_class_nums[ypred] += 1
                                dicts.append((y, ypred))
                                if y == ypred:
                                    counter[y] += 1
                                    all_cor += 1
                        output_filename = '../save_{}_{}_{}_CW/val/val_index_{}_y_ypre_{}'.format(FLAGS.batch_size, data.nHei, data.nWid, val_index, i)
                        with open(output_filename, 'wb') as fo:
                            np.save(fo, dicts)

                        recalls = []
                        precisions = []
                        for k in range(hps.num_classes):
                            if class_nums[k] == 0:
                                recall = 0
                            else:
                                recall = counter[k]/class_nums[k]
                            recalls.append(recall)
                            if pre_class_nums[k] == 0:
                                precision = 0
                            else:
                                precision = counter[k]/pre_class_nums[k]
                            precisions.append(precision)

                            #tf.logging.info('%s: Recall of class %d is: %.4f' % (datetime.now(), k, recall))
                            tf.logging.info('%s: Validation Precision of class %d is: %.4f' % (datetime.now(), k, precision))
                        accuracy = (all_cor*1.0)/all_imgs
                        tf.logging.info('%s: Validation Acc of all classes is %.4f' % (datetime.now(), accuracy))

                        output_filename = '../save_{}_{}_{}_CW/val/val_index_{}_recall_precision_{}.csv'.format(FLAGS.batch_size, data.nHei, data.nWid, val_index, i)
                        col_names = ['recall', 'precision', 'accuracy']
                        acc = [accuracy] * len(precisions)
                        output = {'recall': recalls, 'precision': precisions, 'accuracy': acc}
                        df = pd.DataFrame(output, columns=col_names)
                        with open(output_filename, 'w', newline="") as fo:
                            df.to_csv(fo)

                    saver.save(sess, os.path.join(train_model, 'model'), global_step=i+1)



def evaluate(hps):
    """evaluate"""
    pass


def main(argv=None):
    """Load data and run train"""

    #hps = model.HParams(model=FLAGS.model,
    #                    batch_size=FLAGS.batch_size,
    #                    feature_size=FLAGS.feature_size,
    #                     num_classes=FLAGS.num_classes,
    #                     min_lrn_rate=0.0001,
    #                     lrn_rate=0.1,
    #                     num_residual_units=5,
    #                     use_bottleneck=False,
    #                     weight_decay_rate=0.0002,
    #                     relu_leakiness=0.1,
    #                     optimizer='mom')

    #weight_sample = [1113,6705,514,327,1099,115,142]
    #weight_sample = np.array([1113,6705,514,327,1099,115,142])/10015
    weight_sample_ = [6.02425876, 1.0, 13.04474708, 20.50458716, 6.10100091, 58.30434783, 47.21830986]

    hps = resNet.HParams(
                         feature_size=FLAGS.feature_size,
                         batch_size=FLAGS.batch_size,
                         num_classes=7,
                         #min_lrn_rate=0.0001,
                         #lrn_rate=0.1,
                         num_residual_units=[1,3,4,6,3],
                         use_bottleneck=False,
                         weight_decay_rate=0.0002,
                         relu_leakiness=0.1,
                         optimizer='mom',
                         weight_sample=weight_sample_,
                         use_weight_sample=True)

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
    elif FLAGS.mode == 'test':
        timer_ = timer.timer()
        evaluate(hps)
        timer_.get_duration()


if __name__ == '__main__':
    coloredlogs.install(level='INFO')
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
