"""Code for accout and save some information about predictions"""

import numpy as np
import tensorflow as tf
import pandas as pd

class statistics():
    def __init__(self, hps, epoch, mode='train'):
        """
        Args
            mode: train or test
        """

        self.hps = hps
        self.mode = mode
        self.counter = [ 0 for _ in range(hps.num_classes)]
        self.per_class_nums = [ 0 for _ in range(hps.num_classes)]
        self.triples = []
        #self.class_nums = [ 0 for _ in range(hps.num_classes)]
        self.all_imgs = 0.0
        self.all_cor = 0.0
        self.epoch = epoch

    def add_labels_predictions(self, labels, predictions, path):
        """add batch by batch including labels, predictions, path
        Args
            labels: [batch, num_classes]
            predictions: [batch, num_classes]
            path: [batch]. image name
        """
        labels = np.argmax(labels, axis=1)
        predictions = np.argmax(predictions, axis=1)
        for y, ypred in zip(labels, predictions):
            # precision
            per_class_nums[ypred] += 1
            if y == ypred:
                counter[y] += 1
            self.triples.append((path, y, ypred))

    def get_print_precision(self):
        precisions = []
        for k in range(self.hps.num_classes):
            if self.per_class_nums[k] == 0:
                precision = 0
            else:
                precision = self.counter[k]/self.per_class_nums[k]
            tf.logging.info('%s: %s Precision of class %d is: %.4f' % (datetime.now(), self.mode, k, precision))
            precisions.append(precision)

        return precisions

    def get_acc_normal(self):
        """ predictions_right / all_images
        """
        acc = sum(self.counter)/sum(self.per_class_nums)
        tf.logging.info('%s: %s norm_Acc of all classes is %.4f' % (datetime.now(), self.mode, acc))
        return acc


    def get_acc_imbalanced(self):
        """ precision_add / hps.num_classes
        """
        acc = sum(np.array(self.counter)/np.array(self.per_class_num))/6
        tf.logging.info('%s: %s im_Acc of all classes is %.4f' % (datetime.now(), self.mode, acc))
        return acc

    #def save_statistic(self, filename, data, epoch):
    #    """save statistics as csv, add row by row
    #       row index is the train epoch
    #    """

    #    with open(output_filename, 'a', newline="") as fo:
    #        df.to_csv(fo)

    def save_triples(self, filename):
        """wrap save triples
        Args:
        """
        # triple order is : path, predictions, groundtruth
        self._save_triples(filename, self.triples, self.epoch)

    def _save_triples(self, filename, data, epoch)
        """save triples as csv, add col by col, col name is the epoch name
        Args
            data: []
        """
        with open(output_filename, 'r') as fo:
            df = pd.read_csv(fo)

        # TODO match dict, path to header order
        # TODO add prefix before data_col
        df_add = pd.DataFrame(data, index=[str(epoch), 'predictions', 'ground_truth'])

        df = df.append(df_add)
        with open(output_filename, 'a', newline="") as fo:
            df.to_csv(fo)
        tf.logging.info('Successfully save triples')


