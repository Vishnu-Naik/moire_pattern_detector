"""
Preprocessing
"""
import numpy as np
import cv2
import os
from PIL import Image
import logging

from preprocessing.feature_extractors.local_binary_patterns import LocalBinaryPatterns
from preprocessing.feature_extractors.difference_histograms import DifferenceHistograms
from preprocessing.feature_extractors.colour_features import ColourFeatures
from utils import helper_functions as hf

logger = hf.setup_logger('preprocessing', logging.DEBUG)


def get_multi_feature_for_single_image(image_path: str) -> tuple | None:
    """
    This method returns the multiple features for a single image.
    :param image_path: The path of the image.
    :return: The multiple features for a single image.
    """
    try:
        image = Image.open(image_path)
    except IOError:
        print(f'Error: Couldnt read the file {image_path}. \
            Make sure only images are present in the folder')
        return None

    # making dimensions for all the images uniform
    width, height = image.size
    if height > width:
        image = image.resize((750, 1000))
    else:
        image = image.resize((1000, 750))

    image = np.array(image)

    logger.debug('Getting difference histogram features for the image.')
    difference_hist_helper = DifferenceHistograms()
    difference_hist_feature = difference_hist_helper.get_difference_features(image)

    # radius and num point are predefined
    logger.debug('Getting texture features for the image.')
    desc = LocalBinaryPatterns(24, 8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture_feature = desc.describe(gray_image)

    logger.debug('Getting colour moments features for the image.')
    colour_moments_helper = ColourFeatures()
    color_moments_feature = colour_moments_helper.get_colour_features(image)

    return difference_hist_feature, texture_feature, color_moments_feature


def get_input_features(train_image_dir_path: str):
    """
    This method returns the input features for all the images in the directory path provided.
    :param train_image_dir_path: The path of the directory containing the images.
    :return: The input features for all the images in the directory path provided.
    """
    logger.info('Getting input features for all the images:')
    print('Getting input features for all the images...')
    print(f"Directory path: {train_image_dir_path}")
    difference_hist_features = []
    texture_features = []
    color_moments_features = []
    input_features = []
    for image_path in os.listdir(train_image_dir_path):
        logger.debug(f'Calculating input features for {image_path}')
        difference_hist_feature, texture_feature, color_moments_feature = \
            get_multi_feature_for_single_image(os.path.join(train_image_dir_path, image_path))
        logger.debug(f'Packaging the features for {image_path}')
        difference_hist_features.append(difference_hist_feature.flatten())
        texture_features.append(texture_feature.flatten())
        color_moments_features.append(color_moments_feature.flatten())

    logger.debug('Converting the features to numpy array.')
    difference_hist_features = np.array(difference_hist_features)
    texture_features = np.array(texture_features)
    color_moments_features = np.array(color_moments_features)
    logger.debug('Packing the features of all the images.')
    input_features = np.concatenate((difference_hist_features, texture_features, color_moments_features), axis=1)

    return input_features
