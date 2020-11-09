import cv2


def transform(image_array):
    """
    Simple transformation to histogram equalization
    :param image_array:
    :return: transformed array
    """
    equalizer = cv2.createCLAHE(1.0, (3, 3))
    image = equalizer.apply(image_array)
    image = cv2.fastNlMeansDenoising(image)
    return image
