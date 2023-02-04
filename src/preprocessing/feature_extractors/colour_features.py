import numpy as np
import cv2
import logging

from utils import helper_functions as hf

logger = hf.setup_logger('preprocessing', logging.DEBUG)


class ColourFeatures:
    """
    This class is used to colour features from the input image.
    This class has the following methods:
        1. get_moments: This method returns the spatial moments of the image.
        2. get_color_moments: This method returns the color moments of the image.
        3. get_colour_features: This method returns a dictionary of the features extracted from the image.
    """

    @staticmethod
    def get_moments(moments: dict, order: int) -> list:
        """
        This method returns the spatial moments of the image.
        :param moments: The dictionary of the moments of the image.
        :param order: The order of the moment.
        :return: A list of the spatial moments of the image.
        """
        keys = moments.keys()
        spatials_moments = []
        for key in keys:
            if 'm' in key and len(key) == 3:
                moment_index = key.split('m')[-1]
                moment_order = int(moment_index[0]) + int(moment_index[1])
                if moment_order == order:
                    spatials_moments.append(moments[key])
        return spatials_moments

    @staticmethod
    def get_component_moments(component: np.ndarray, num_orders: int) -> list:
        """
        This method returns the color moments of the image.
        :param component: The image component.
        :param num_orders: The number of orders of the moment.
        :return: A list of the color moments of an image component.
        """
        moments_dic = cv2.moments(component)
        component_moments = []
        for order_index in range(1, num_orders + 1):
            temp_moments = ColourFeatures.get_moments(moments_dic, order_index)
            component_moments.append(temp_moments)
        return component_moments

    @staticmethod
    def get_color_moments(image_array: np.ndarray) -> list:
        """
        This method returns the color moments of the all the components of the image.
        :param image_array: The input image.
        :return: A list of the color moments.
        """
        image_channels = cv2.split(image_array)
        rgb_moments = []
        for color_component in image_channels:
            comp_moments = ColourFeatures.get_component_moments(color_component, 3)

            balanced_array = np.zeros([len(comp_moments), len(max(comp_moments, key=lambda x: len(x)))])
            for i, j in enumerate(comp_moments):
                balanced_array[i][0:len(j)] = j
            rgb_moments.append(balanced_array)
        return rgb_moments

    def get_colour_features(self, image_array: np.ndarray) -> list:
        """
        This method returns a dictionary of the features extracted from the image.
        :param image_array: The input image.
        :return: A list of the features extracted from the image.
        """
        hsv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)

        contrast_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        bgr_moments_feature = self.get_color_moments(image_array)
        hsv_moments_feature = self.get_color_moments(hsv_image)
        contrast_moments_feature = self.get_color_moments(contrast_image)

        contrast_feature = contrast_image.std()

        temp_color_moments_feature = bgr_moments_feature + hsv_moments_feature + contrast_moments_feature

        temp_color_moments_feature = np.array(temp_color_moments_feature)

        temp_color_moments_feature = temp_color_moments_feature.reshape(21, -1)
        return temp_color_moments_feature
