import numpy as np
from skimage import measure, morphology, segmentation

from utils.logging import logger


def segment_from_image(
    image_array: np.ndarray, erosion=1, closing=4, dilation=16
):
    """
    Segment critical parts from image array
    :param image_array: Image in numpy array form
    :param erosion: erosion connectivity factor
    :param closing: closing connectivity factor
    :param dilation: dilation connectivity factor
    :return: transformed image array
    """
    # Assert that image is of stipulated shape
    try:
        assert len(image_array.shape) == 3 and image_array.shape[2] == 3
    except AssertionError:
        logger.error("Image array must be of shape (?, ?, 3)")

    # Type casting as int and create binary mask using mean value
    image_array = image_array.astype("uint8")
    image_array_binary_mask = image_array < image_array.mean()

    # Clear binary mask boarder
    for channel in range(image_array_binary_mask.shape[2]):
        image_array_binary_mask[:, :, channel] = segmentation.clear_border(
            image_array_binary_mask[:, :, channel]
        )

    # Label the image and dim the binary mask according to labelled region
    labelled_image = measure.label(image_array_binary_mask)
    regions = measure.regionprops(labelled_image)
    areas = [(region.area, region.label) for region in regions]
    areas.sort()

    if len(areas) > 2:
        max_area = areas[-2][0]
        for region in regions:
            if region.area < max_area:
                for channel in region.coords:
                    image_array_binary_mask[
                        channel[0], channel[1], channel[2]
                    ] = 0

    # Additional CV steps to further segment core areas from the rest of the image
    image_array_binary_mask = morphology.binary_erosion(
        image_array_binary_mask, selem=np.ones((erosion,) * 3)
    )
    image_array_binary_mask = morphology.binary_closing(
        image_array_binary_mask, selem=np.ones((closing,) * 3)
    )
    image_array_binary_mask = morphology.binary_dilation(
        image_array_binary_mask, selem=np.ones((dilation,) * 3)
    )

    return np.multiply(image_array_binary_mask, image_array)
