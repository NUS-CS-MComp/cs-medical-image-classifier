from typing import Any

import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu
from skimage.restoration import denoise_nl_means, estimate_sigma

from utils.logging import logger


def transform(image_array: np.ndarray):
    """
    Rescale intensity of given image array
    :param image_array: image in numpy array form
    :return: transformed image array
    """
    # Assert that image is of stipulated shape
    try:
        assert len(image_array.shape) == 3 and image_array.shape[2] == 3
    except AssertionError:
        logger.error("Image array must be of shape (?, ?, 3)")

    # Convert to grayscale
    image_array = image_array.astype("uint8")
    grayscale = rgb2gray(image_array)

    # Rescale intensity
    lower, upper = np.percentile(grayscale, (25, 90))
    rescaled: Any = rescale_intensity(grayscale, in_range=(lower, upper))

    # Denoising
    sigma_est = np.mean(estimate_sigma(rescaled))
    denoised = denoise_nl_means(rescaled, h=0.8 * sigma_est, sigma=sigma_est)

    # Binary thresholding
    binary = denoised > threshold_otsu(denoised)
    final = np.multiply(binary, rescaled)

    return np.uint8(final * 255)
