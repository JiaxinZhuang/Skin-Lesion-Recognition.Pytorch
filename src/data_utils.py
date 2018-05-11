"""data_utils
    used to process and analyse ISIC-2018 data
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import cv2 as cv
from matplotlib import pyplot as plt
import logging
import os


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("ISIC2018_Task3_Training_GroundTruth",
                       "ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv",
                       """Name of Task3_Training_GroundTruth csv file""")
tf.flags.DEFINE_string("ISIC2018_Task3_Training_Input",
                        "ISIC2018_Task3_Training_Input",
                        """Name of ISIC2018_Task3_Training_Input directory""")

logging.basicConfig(level=logging.INFO)

DATA_URL='../data'

class input_data:
    """input data including training input and ground truth
    """
    def __init__(self):
        self.ISIC2018_Task3_Training_Input_path = os.path.join(DATA_URL, FLAGS.ISIC2018_Task3_Training_Input)
        self.ISIC2018_Task3_Training_GroundTruth_path = os.path.join(DATA_URL,
                FLAGS.ISIC2018_Task3_Training_GroundTruth)
        self.batch_size = 128
        logging.info('ISIC2018_Task3_Training_Input %s' % self.ISIC2018_Task3_Training_Input_path)
        logging.info('ISIC2018_Task3_Training_GroundTruth %s' % self.ISIC2018_Task3_Training_GroundTruth_path)


    def load_image_with_all(self, directory, images_path):
        # add prefix path
        images_path_with_prefix = map(lambda x: os.path.join(directory, x), images_path)
        images_path_with_prefix = list(images_path_with_prefix)
        self.ISIC2018_Task3_Training_Input_nps = []

        for filename, filepath in zip(images_path, images_path_with_prefix):
            img = cv.imread(filepath)
            self.ISIC2018_Task3_Training_Input_nps.append(img)
            #img = cv.imread(filepath, 0)
            #img_blur = cv.medianBlur(img,5)
            #processed_img = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
            #cv.imwrite(filename, processed_img)
            #cv.imwrite('_' + filename, img)
            #break

    def get_images_path_from_directory(self, directory):
        logging.info('Start loading images from directory %s' % directory)
        images_path = os.listdir(directory)
        # get rid of unrelated file
        images_path = filter(lambda x: x.split('.')[-1] == 'jpg', images_path)
        images_path = list(images_path)

        return images_path

    def get_np_from_csv(self, filenmae, header=0):
        """
            filename
            header: start from first line of the csvfile
        """
        csvfile = pd.read_csv(filename, header=header)
        filename_np = csvfile.values
        return filename_np


if __name__=='__main__':
    data = input_data()
    images_list = data.get_images_path_from_directory(data.ISIC2018_Task3_Training_Input_path)
    data.load_image_with_all(data.ISIC2018_Task3_Training_Input_path, images_list)
