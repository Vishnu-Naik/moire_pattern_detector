from ast import Dict
import itertools
import numpy as np
import cv2
from scipy import signal
import logging
from utils import helper_functions as hf

logger = hf.setup_logger('preprocessing', logging.DEBUG)


class DifferenceHistograms:
    """
    Difference Histograms

    This class is used to extract features from the input image.
    The features are extracted using difference of varied filters.
    The filters used are:
        1. Horizontal
        2. Vertical
        3. Diagonal
        4. Anti-diagonal
    The above filters are used in multiple combinations to extract features.
    The features are extracted from the image and stored in a dictionary.
    This class has the following methods:
        1. get_kernel_bank: This method returns a dictionary of the filters used.
        2. get_feature_array: This method returns a numpy array of the features extracted from the image.
        3. get_histogram: This method returns a numpy array of the histogram of the features extracted from the image.
        4. get_norm_histogram: This method returns a numpy array of the normalized histogram of the features extracted
                                from the image.
        5. get_features: This method returns a dictionary of the features extracted from the image.
    """
    features = {
        'name': [],
        'feature_array': [],
        'histogram': [],
        'norm_histogram': []
    }

    def __init__(self) -> None:
        self.kernel_bank = self.get_kernel_bank()

    @staticmethod
    def get_kernel_bank() -> Dict:
        """
        This method returns a dictionary of the filters used.
        :return: Dictionary of the filters as shown below:
            {
                "horizontal_filter": horizontal_filter,
                "verticle_filter": verticle_filter,
                "diagonal_filter": diagonal_filter,
                "antidiagonal_filter": antidiagonal_filter
            }
        """
        horizontal_filter = np.array((
            [[1, -1]]), dtype="int")
        verticle_filter = np.array((
            [1],
            [-1]), dtype="int")
        diagonal_filter = np.array((
            [1, 0],
            [0, -1]), dtype="int")
        antidiagonal_filter = np.array((
            [0, 1],
            [-1, 0]), dtype="int")

        kernel_bank = {
            "horizontal_filter": horizontal_filter,
            "verticle_filter": verticle_filter,
            "diagonal_filter": diagonal_filter,
            "antidiagonal_filter": antidiagonal_filter
        }

        return kernel_bank

    @staticmethod
    def get_difference_images_for_channel(channel_data, kernel_bank) -> tuple:
        """
        This method returns a tuple of the features extracted from the image.
        :param channel_data: Single channel data from the image (eg. Blue, Green or Red).
        :param kernel_bank: The dictionary of the filters used.
        :return: A tuple of a list of two combination of kernels and
                the features extracted by applying kernels in combination of 2 on the channel data.
        """

        name_list = []
        feature_array = []
        for (kernel_name, kernel) in kernel_bank.items():
            logger.debug(f"[INFO] applying {kernel_name} kernel")
            filtered_output = signal.convolve2d(channel_data, kernel, boundary='symm', mode='same')
            name_list.append(kernel_name[0])
            feature_array.append(filtered_output)

        filter_combinations = []
        filter_iter = itertools.combinations_with_replacement([i for i in kernel_bank.keys()], 2)
        for i in filter_iter:
            filter_combinations.append(i)

        num_images = 0
        for filter_comb in filter_combinations:
            intermediate_output = signal.convolve2d(
                channel_data,
                kernel_bank[filter_comb[0]],
                boundary='symm',
                mode='same')
            filtered_output = signal.convolve2d(
                intermediate_output,
                kernel_bank[filter_comb[1]],
                boundary='symm',
                mode='same')
            num_images = num_images + 1
            logger.debug(f'Image {num_images} of {len(filter_combinations)}')
            logger.debug(f'[INFO] applying {filter_comb[0]} and {filter_comb[1]}')
            logger.debug(f'[INFO] dimension {filtered_output.shape}')
            name_list.append(filter_comb[0][0] + filter_comb[1][0])
            feature_array.append(filtered_output)

        return name_list, feature_array

    @staticmethod
    def get_histogram_and_norm(feature_array) -> tuple:
        """
        This method returns a tuple of the histogram and normalized histogram of the features extracted from the image.
        :param feature_array: The numpy array of the features extracted from the image.
        :return: A tuple of the histogram and normalized histogram of the features extracted from the image.
        """
        hist_array = []
        norm_hist = []
        for feature_data in feature_array:
            hist, _ = np.histogram(feature_data, list(range(-100, 110, 10)))
            hist_array.append(hist)
            hist = hist / (750 * 1000)
            norm_hist.append(hist)

        return hist_array, norm_hist

    @staticmethod
    def get_translated_histogram(original_hist, mean_value):
        """
        This method returns a numpy array of the translated histogram of the histogram extracted from the image.
        :param original_hist: The numpy array of the histogram of the features extracted from the image.
        :param mean_value: The mean value around which the histogram has to be translated.
        :return: A numpy array of the translated histogram of the histogram extracted from the image.
        """
        temp_hist_plus = original_hist + mean_value
        temp_hist_minus = original_hist - mean_value
        translated_hist = (temp_hist_plus + temp_hist_minus) / 2
        return translated_hist

    @staticmethod
    def get_difference_histogram_feature(translate_histogram_function,
                                         feature_array,
                                         difference_histogram_dims):
        """
        This method returns a numpy array of the difference histogram of the histogram extracted from the image.
        :param translate_histogram_function: The function to translate the histogram.
        :param feature_array: The numpy array of the histogram of the features extracted from the image.
        :param difference_histogram_dims: The number of dimensions of the difference histogram.
        :return: A numpy array of the difference histogram of the histogram extracted from the image.
        """
        translated_histogram = [feature_array]
        for mean_value in range(1, difference_histogram_dims):
            temp_hist = translate_histogram_function(feature_array, mean_value + 1)
            translated_histogram.append(temp_hist)
        return translated_histogram

    def get_difference_features(self, image_array: np.ndarray):
        """
        This method returns a dictionary of the features extracted from the image.
        :param image_array: The numpy array of the image.
        :return: An array of the features extracted from the image.
        """
        image_channels = cv2.split(image_array)

        _, _, r_comp_image = image_channels

        name_list, feature_array = \
            self.get_difference_images_for_channel(r_comp_image, self.kernel_bank)
        self.features['name'] = name_list
        self.features['feature_array'] = feature_array

        # ## Histogram for all difference images
        histogram, norm_histogram = self.get_histogram_and_norm(self.features['feature_array'])
        self.features['histogram'] = histogram
        self.features['norm_histogram'] = norm_histogram

        # ## Deriving multi mean histograms

        temp_difference_hist_feature = []
        for norm_hist in self.features['norm_histogram']:
            hist_feature = self.get_difference_histogram_feature(
                self.get_translated_histogram,
                norm_hist,
                7)
            temp_difference_hist_feature = temp_difference_hist_feature + hist_feature

        temp_difference_hist_feature = temp_difference_hist_feature + self.features['norm_histogram']

        return np.array(temp_difference_hist_feature)
