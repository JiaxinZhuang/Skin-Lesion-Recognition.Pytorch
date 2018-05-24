import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

def gray_world(img):
    """
    Parameters
    ----------
    img: 2D numpy array
        The original image with format of (h, w, c)
    """
    img = img.astype(np.float)
    pixels_num = img.shape[0] * img.shape[1]
    bgr_avg = np.sum(np.sum(img, axis = 0), axis = 0) / pixels_num
    gray_avg = np.sum(bgr_avg) / img.shape[2]
    bgr_k = gray_avg / bgr_avg
    return img * np.transpose(bgr_k)
def white_patch_retinex(img, para):
    """
    Parameters
    ----------
    img: 2D numpy array
        The original image with format of (h, w, c)
    para: 0 or 1
        0 : original algorithm
        1 : optimization algorithm
    """
    img = img.astype(np.float)
    out = np.zeros(img.shape, dtype=float)
    L = [0, 0, 0]
    if para == 0:
        for i in range(3):
            L[i] = np.max(img[:, :, i].flatten())
            out[:, :, i] = img[:, :, i] * 255.0 / L[i]
    elif para == 1:
        n_p = 0.1 * img.shape[0] * img.shape[1]
        for i in range(3):
            H, bins = np.histogram(img[:, :, i].flatten(), 256)
            sums = 0
            for j in range(255, -1, -1):
                if sums < n_p:
                    sums += H[j]
                else:
                    L[i] = j
                    out[:, :, i] = img[:, :, i] * 255.0 / L[i]
                    break
    return out
def shade_of_gray(img, power=6, gamma=None):
    """
    Parameters
    ----------
    img: 2D numpy array
        The original image with format of (h, w, c)
    power: int
        The degree of norm, 6 is used in reference paper
    gamma: float
        The value of gamma correction, 2.2 is used in reference paper
    """
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256,1), dtype='uint8') * 0
        for i in xrange(256):
            look_up_table[i][0] = 255 * pow(i/255, 1/gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0,1)), 1/power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1/(rgb_vec*np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    return img.astype(img_dtype)
def general_gray_world(img, power=2, sigma=3):
    """
    Parameters
    ----------
    img: 2D numpy array
        The original image with format of (h, w, c)
    power: int
        The degree of norm, 2 is used in reference paper
    sigma: float
        Filtering image with a Guassian low-pass filter with standard deviation sigma
    """
    blur = cv2.GaussianBlur(img, (3, 3), sigma)
    return shade_of_gray(blur, power)
