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

    def process_images_with_threshold_from_gray(self, images_nps, images_path):

        for eimage_name, eimage_np in zip(images_path, images_nps):
            img_blur = cv.medianBlur(eimage_np,5)
            processed_img = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
            combined = np.concatenate((processed_img, eimage_np), axis=0)
            assert processed_img.shape[0] == eimage_np.shape[0]
            assert processed_img.shape[1] == eimage_np.shape[1]
            print(eimage_name)
            cv.imwrite(eimage_name, combined)

    def get_disease_area_information_from_rgb(self, images_nps):
        """get_disease_area_information_from_rgb
            using k-means which k is 2 to extract disease area from whole image
            here we get a triple information from each image

            inputs:
                images_nps: a list containing many images in numpy with rgb mode
                        [images_np, ...]
                        images_np.shape = (row, col, 3)
            outputs:
                a list with same numbers as images_nps
                [(images_processed_as_one_and_zero, disease_area_nums, pixels_num),
                ...]
        """
        output_list = []
        from sklearn.cluster import KMeans
        for images_np in images_nps:
            kmeans = KMeans(n_clusters=2, random_state=0)
            images_np_vector = np.reshape(images_np, (-1, 3))
            kmeans.fit(images_np_vector)
            images_processed_as_one_and_zero = np.reshape(kmeans.labels_,
                    (images_np.shape[:2]))
            print(images_processed_as_one_and_zero.shape)
            disease_area_nums = np.sum(images_processed_as_one_and_zero.flatten())
            pixels_num = images_np.shape[0] * images_np.shape[1]
            # Assuming that valid area is less than 50% of image
            # Numbers of valid area are all ones, and others are zones.
            threshold = 0.5
            if disease_area_nums >= threshold * pixels_num:
                images_processed_as_one_and_zero = np.ones(images_processed_as_one_and_zero.shape, dtype=int) - images_processed_as_one_and_zero
                disease_area_nums = pixels_num - disease_area_nums
            triple_list = (images_processed_as_one_and_zero, disease_area_nums, pixels_num)
            output_list.append(triple_list)
        return output_list

    # TODO
    def __visual_diease_area_one_image__(self, image_np):
        """visual_diease_area_one_image
            visual disease area, and show it to help debug

            inputs:
                image_np: a image numpy from get_disease_area_information_from_rgb
            output:
                no
        """


    # TODO
    def generate_save_disease_area_information_as_images(self, images_nps, save_directory):
        """generate_save_disease_area_information_as_images
            using cv to generate binary picture and concatenate corresponding
            images in horizontal way and save them under save_directory
        """
        images_processed_as_one_and_zero_batch = get_disease_area_information_from_rgb(images_nps)[0]
        for idx, image_np in enumerate(images_np):
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(image_np)
            plt.subplot(1, 2, 2)
            plt.imshow(images_processed_as_one_and_zero_batch[i], 'Greys')
            plt.savefig(save_directory)
            plt.close()

    def load_image_with_all(self, directory, images_path, gray=True):
        # add prefix path
        images_path_with_prefix = map(lambda x: os.path.join(directory, x), images_path)
        images_path_with_prefix = list(images_path_with_prefix)
        output = []

        for filename, filepath in zip(images_path, images_path_with_prefix):
            if gray == True:
                img = cv.imread(filepath, 0)
            else:
                img = cv.imread(filepath)
            output.append(img)
        return output

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
    imgs = data.load_image_with_all(data.ISIC2018_Task3_Training_Input_path, images_list, gray=False)
    #data.process_images_with_threshold_from_gray(imgs, images_list)
    data.get_disease_area_information_from_rgb(imgs)
