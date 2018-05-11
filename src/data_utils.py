"""data_utils
    used to process and analyse ISIC-2018 data
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import os


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("ISIC2018_Task3_Training_GroundTruth",
                       "ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv",
                       """Name of Task3_Training_GroundTruth csv file""")
tf.flags.DEFINE_string("ISIC2018_Task3_Training_Input",
                        "ISIC2018_Task3_Training_Input",
                        """Name of ISIC2018_Task3_Training_Input directory""")

logging.basicConfig(logging.INFO)

DATA_URL='../data'

class input_data:
    """input data including training input and ground truth
    """
    def __init__(self):
        self.ISIC2018_Task3_Training_Input_path = os.path.join(DATA_URL, FLAGS.ISIC2018_Task3_Training_Input)
        self.ISIC2018_Task3_Training_GroundTruth_path = os.path.join(DATA_URL,
                FLAGS.ISIC2018_Task3_Training_GroundTruth)
        logging.info('ISIC2018_Task3_Training_Input %s' % self.ISIC2018_Task3_Training_Input)
        logging.info('ISIC2018_Task3_Training_GroundTruth %s' % self.ISIC2018_Task3_Training_GroundTruth)

    def get_images_path_from_directory(self, directory):
        logging.info('Start loading images from directory %s' % FLAGS.ISIC2018_Task3_Training_Input)
        images_path = []
        for root, _, files in os.walk(directory):
            if files.split('.')[-1] != 'jpg':
                images_path.append(os.path.join(root, files))
        return images_path

    def get_np_from_csv(self, filenmae, header=0):
        """
            filename
            header: start from first line of the csvfile
        """
        csvfile = pd.read_csv(filename, header=header)
        filename_np = csvfile.values
        return filename_np



