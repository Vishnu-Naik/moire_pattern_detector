import numpy as np
from skimage import feature
import logging

from utils import helper_functions as hf

logger = hf.setup_logger('preprocessing', logging.DEBUG)


# # Texture feature
# This program is a direct adaptation from the article
# (https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/)

class LocalBinaryPatterns:
    """
    Local Binary Patterns
    This class is used to extract spatial features using Local Binary Search from the input image.

    This class has the following methods:
        1. describe: This method returns a numpy array of the features extracted from the image.
    """
    def __init__(self, num_points, radius):
        # store the number of points and radius
        self.num_points = num_points
        self.radius = radius

    def describe(self, image_array, eps=1e-7):
        """
        compute the Local Binary Pattern representation
        of the image, and then use the LBP representation
        to build the histogram of patterns
        :param image_array: input image
        :param eps: epsilon value
        :return: numpy array of the features extracted from the image
        """
        logger.debug("Computing Local Binary Pattern")
        lbp = feature.local_binary_pattern(image_array, self.num_points,
                                           self.radius, method="uniform")
        (lsb_hist, _) = np.histogram(lbp.ravel(),
                                     bins=np.arange(0, self.num_points + 3),
                                     range=(0, self.num_points + 2))
        # normalize the histogram
        lsb_hist = lsb_hist.astype("float")
        lsb_hist /= (lsb_hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        logger.debug("Local Binary Pattern computed")
        return np.array(lsb_hist)
