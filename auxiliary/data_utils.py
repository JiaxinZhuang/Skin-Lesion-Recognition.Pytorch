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
import process_bar
import timer


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

    #def process_images_with_threshold_from_gray(self, images_nps, images_path):

    #    for eimage_name, eimage_np in zip(images_path, images_nps):
    #        img_blur = cv.medianBlur(eimage_np,5)
    #        processed_img = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
    #        combined = np.concatenate((processed_img, eimage_np), axis=0)
    #        assert processed_img.shape[0] == eimage_np.shape[0]
    #        assert processed_img.shape[1] == eimage_np.shape[1]
    #        print(eimage_name)
    #        cv.imwrite(eimage_name, combined)

    def get_ground_truth(self, filename):
        ground_truth = self.get_np_from_csv(filename)
        return ground_truth

    def get_disease_area_information_from_rgb_by_batch(self, images_nps, batch_size):
        """get_disease_area_information_from_rgb
            using k-means which k is 2 to extract disease area from whole image
            here we get a triple information from each image

            inputs:
                images_nps: a list containing many images in numpy with rgb mode
                        [images_np, ...]
                        images_np.shape = (row, col, 3)
                batch_size
            outputs:
                a list with same numbers as images_nps
                [(images_processed_as_one_and_zero, disease_area_nums, pixels_num),
                ...]
        """
        output_list = []
        logging.info('get disease area information from rgb')
        process_bar_ = process_bar.process_bar(len(images_nps))
        from sklearn.cluster import KMeans
        count = 0
        for images_np in images_nps:
            count = count + 1
            process_bar_.show_process()
            kmeans = KMeans(n_clusters=2, random_state=0)
            images_np_vector = np.reshape(images_np, (-1, 3))
            kmeans.fit(images_np_vector)
            images_processed_as_one_and_zero = np.reshape(kmeans.labels_,
                    (images_np.shape[:2]))
            disease_area_nums = np.sum(images_processed_as_one_and_zero.flatten())
            pixels_num = images_np.shape[0] * images_np.shape[1]
            # Assuming that valid area is less than 50% of image
            # Numbers of valid area are all ones, and others are zones.
            threshold = 0.5
            if disease_area_nums >= threshold * pixels_num:
                images_processed_as_one_and_zero = np.ones(images_processed_as_one_and_zero.shape, dtype=int) - images_processed_as_one_and_zero
                disease_area_nums = pixels_num - disease_area_nums
            triple_list = [images_processed_as_one_and_zero, disease_area_nums, pixels_num]
            output_list.append(triple_list)
            if count >= batch_size:
                yield output_list
                output_list = []
                count = 0
        if count > 0:
            yield output_list
        #return output_list

    # TODO
    def __visual_diease_area_one_image__(self, image_np):
        """visual_diease_area_one_image
            visual disease area, and show it to help debug

            inputs:
                image_np: a image numpy from get_disease_area_information_from_rgb
            output:
                no
        """
        plt.figure()
        plt.plot(image_np)
        plt.show()
        plt.close()

    def save_disease_area_information_as_images(self, pro_images_nps,images_nps, images_path, save_directory):
        """generate_save_disease_area_information_as_images
            using cv to generate binary picture and concatenate corresponding
            images in horizontal way and save them under save_directory
        """
        logging.info('generate save disease area information as images')
        process_bar_ = process_bar.process_bar(len(pro_images_nps))
        for image_name, image_np, pro_image_np in zip(images_path, images_nps, pro_images_nps):
            image_name = os.path.join(save_directory, image_name)
            image_name = image_name.replace('.jpg', '.png')
            image_np = cv.cvtColor(image_np, cv.COLOR_RGB2GRAY)
            pro_image_np = pro_image_np * 255
            #print(image_np)
            #print(pro_image_np)
            combined = np.concatenate((image_np, pro_image_np), axis=0)
            #logging.info('save images_name %s' % image_name)
            cv.imwrite(image_name, combined)
            #cv.waitKey(0)
            process_bar_.show_process()

        #images_processed_as_one_and_zero_batch = get_disease_area_information_from_rgb(images_nps)[0]
        #for idx, image_np in enumerate(images_np):
        #    plt.figure()
        #    plt.subplot(1, 2, 1)
        #    plt.imshow(image_np)
        #    plt.subplot(1, 2, 2)
        #    plt.imshow(images_processed_as_one_and_zero_batch[idx], 'Greys')
        #    plt.savefig(save_directory)
        #    plt.close()

    def load_image_from_pickle(self, filename):
        if not os.path.exists(filename):
            logging.info('No pickle file')
            sys.exit(-1)
        else:
            logging.info('load_image_from_pickle')
            data = self.unpickle(filename)
            images_path = data['images_path']
            images_nps = data['images_nps']
            return images_path, images_nps

    def load_image_from_directory(self, directory, images_path, gray=True):
        # add prefix path
        images_path_with_prefix = map(lambda x: os.path.join(directory, x), images_path)
        images_path_with_prefix = list(images_path_with_prefix)
        output = []

        logging.info("loading image with all")
        process_bar_ = process_bar.process_bar(len(images_path))
        for filename, filepath in zip(images_path, images_path_with_prefix):
            process_bar_.show_process()
            if gray == True:
                img = cv.imread(filepath, 0)
            else:
                img = cv.imread(filepath)
            output.append(img)
        return output

    def unpickle(self, filename):
        import pickle
        with open(filename, 'rb') as fo:
            data = pickle.load(fo)
        return data

    def save_pickle(self, filename, data):
        import pickle
        with open(filename, 'wb') as fo:
            pickle.dump(data, fo)

    def get_images_path_from_directory(self, directory):
        logging.info('Start loading images from directory %s' % directory)
        images_path = os.listdir(directory)
        # get rid of unrelated file
        images_path = filter(lambda x: x.split('.')[-1] == 'jpg', images_path)
        images_path = list(images_path)

        return images_path

    def get_np_from_csv(self, filename, header=0):
        """
            filename
            header: start from first line of the csvfile
        """
        csvfile = pd.read_csv(filename, header=header)
        filename_np = csvfile.values
        return filename_np

    def save_disease_area_csv(self, disease_areas, images_area, images_path,  ground_truth, filename):
        convert_vec = [0,1,2,3,4,5,6]
        images_path = list(map(lambda x: x.split('.')[0], images_path))
        ground_truth_ = []
        for i in images_path:
            raw_label_vec = list(filter(lambda x:x[0] == i, ground_truth))
            assert len(raw_label_vec) == 1
            label_vec = raw_label_vec[0][1:]
            assert len(label_vec) == 7
            label = np.dot(convert_vec, label_vec)
            ground_truth_.append(label)
        assert len(disease_areas) == len(images_area)
        assert len(ground_truth_) == len(images_path)
        data = {'images_path': images_path, 'disease_areas': disease_areas, 'images_area': images_area, 'disease_areas_ratio': np.array(disease_areas)/np.array(images_area), 'ground_truth': ground_truth_}
        pd.DataFrame(data).to_csv(filename)

    def generate_and_save_pickle(self, train_data_path):
        logging.info('generate_and_save_pickle %s' % train_data_path)
        images_path = self.get_images_path_from_directory(train_data_path)
        images_nps = self.load_image_from_directory(images_path, images, gray=False)
        dicts = {'images_path': images_nps, 'images_nps': images_nps}
        self.save_pickle(train_data_path + '_pickle', dicts)


if __name__=='__main__':
    timer_ = timer.timer()
    batch_size = 128
    data = input_data()
    # generate images from directory and save as pickle
    #data.generate_and_save_pickle(FLAGS.ISIC2018_Task3_Training_Input)
    # load images from pickle
    ground_truth = data.get_ground_truth(data.ISIC2018_Task3_Training_GroundTruth_path)
    images_path, images_nps = data.load_image_from_pickle(data.ISIC2018_Task3_Training_Input_path+'_pickle')
    output = data.get_disease_area_information_from_rgb_by_batch(images_nps, batch_size)
    count = 0
    disease_areas = []
    images_area = []
    x = next(output)
    while len(x) > 0:
        index = count *  batch_size
        index_end = min((count+1) * batch_size, len(images_path))
        logging.info('\nfrom index %d to index_end %d' % (index, index_end-1))
        pro_images_nps = []
        for xx in x:
            pro_images_nps.append(xx[0])
            disease_areas.append(xx[1])
            images_area.append(xx[2])
        data.save_disease_area_information_as_images(pro_images_nps, images_nps[index:index_end], images_path[index:index_end], '../data/bio_image')
        count = count + 1
        x = next(output)
    data.save_disease_area_csv(disease_areas, images_area, images_path, ground_truth, '../data/bio_disease_areas.csv')
    timer_.get_duration()
