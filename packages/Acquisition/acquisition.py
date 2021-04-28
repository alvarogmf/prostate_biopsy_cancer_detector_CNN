import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import skimage.io
from skimage.transform import resize, rescale
import cv2
from PIL import Imag

def load_csv(path):
    """
    Load the main csv where the information about each image is stored (id, scores...)
    :param path: path where the csv is located
    :param name: desired name to be placed to the dataframe
    :return: a Pandas Dataframe
    """
    csv = pd.read_csv(path)
    return csv

def transform_images(data_dir, save_dir, csv_name, format = '.png'):
    """
    Transform the images from their original format and shape into an easier format to work with.
    Also, the sizes of the images will be reduced to 10% their original size.
    :param load_path:
    :param save_path:
    :param csv_name:
    :param format:
    :return:
    """
    for img_id in tqdm(csv_name.image_id):
         load_path = data_dir + img_id + '.tiff'
         save_path = save_dir + img_id + format

         biopsy = skimage.io.MultiImage(load_path)
         img = cv2.resize(biopsy[-1], (0,0), fx = 0.1, fy = 0.1)
         cv2.imwrite(save_path, img)