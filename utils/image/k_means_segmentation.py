import numpy as np
from skimage import exposure, morphology, measure, restoration
from sklearn.cluster import KMeans


def transform(
    image_array, k_means_cluster=2, boundary_factor=10, only_cropped=True
):
    """
    Transform using K-means segmentation
    from https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/ with small adjustments

    :param image_array: image array
    :param k_means_cluster: number of clusters
    :param boundary_factor: boundary to locate the center
    :param only_cropped: boolean flag for cropping out region only without mask
    :return: transformed image
    """

    # Convert to grayscale and do basic transformation
    image_array = exposure.equalize_adapthist(image_array, clip_limit=0.01)
    image_array = restoration.denoise_bilateral(image_array)
    transformed_image_array = np.copy(image_array)

    # Calculate image features
    row_size = image_array.shape[0]
    col_size = image_array.shape[1]

    (
        left_boundary,
        right_boundary,
    ) = row_size // boundary_factor, row_size // boundary_factor * (
        boundary_factor - 1
    )
    (
        up_boundary,
        down_boundary,
    ) = col_size // boundary_factor, col_size // boundary_factor * (
        boundary_factor - 1
    )

    # Normalize the data
    mean = np.mean(image_array)
    std = np.std(image_array)
    image_array = (image_array - mean) / std

    # Define middle zone
    middle = image_array[
        left_boundary:right_boundary, up_boundary:down_boundary
    ]
    middle_mean = np.mean(middle)
    max_pixel = np.max(image_array)
    min_pixel = np.min(image_array)

    # Offsetting max and min value to mean
    image_array[image_array == max_pixel] = middle_mean
    image_array[image_array == min_pixel] = middle_mean

    # K-means
    k_means = KMeans(n_clusters=k_means_cluster).fit(
        np.reshape(middle, (np.prod(middle.shape), 1))
    )
    centers = sorted(k_means.cluster_centers_.flatten())
    threshold = np.mean(centers)
    mask = np.where(image_array < threshold, 1, 0)

    # Morphological transformation
    eroded = morphology.erosion(mask, np.ones((6, 6)))
    dilation = morphology.dilation(eroded, np.ones((3, 3)))

    # Labelling
    labels = measure.label(dilation)
    labels = morphology.remove_small_objects(labels)
    regions = [
        (region.area, region.label, region.coords, region.bbox)
        for region in measure.regionprops(labels)
    ]
    selected_regions = []

    for region in regions:
        area, coords, box = region[0], region[2], region[3]
        select_current_label = False
        is_valid_shape = (
            (box[2] - box[0]) / row_size <= 1
            and (box[3] - box[1]) / col_size <= 0.75
            and abs(box[2] + box[0] - row_size) / row_size <= 0.5
            and abs(box[3] + box[1] - col_size) / col_size <= 0.75
        )

        if is_valid_shape:
            select_current_label = True
        if select_current_label:
            selected_regions.append((area, box))
        for coord in coords:
            mask[coord[0], coord[1]] = 1 if select_current_label else 0

    # Final output
    pre_final = morphology.dilation(mask, np.ones((3, 3)))
    final = pre_final * image_array

    # Cropping
    regions_bbox = np.array([region[1] for region in selected_regions]).T
    new_bbox = np.concatenate(
        (regions_bbox.min(axis=1)[:2], regions_bbox.max(axis=1)[2:])
    )
    min_r, min_c, max_r, max_c = new_bbox
    width = max_r - min_r
    height = max_c - min_c
    is_even_size = abs(height - width) % 2 == 0

    if height > width:
        if is_even_size:
            min_r -= abs(height - width) // 2
            max_r += abs(height - width) // 2
        else:
            min_r -= (abs(height - width) + 1) // 2
            max_r += (abs(height - width) - 1) // 2
    elif width < height:
        if is_even_size:
            min_c -= abs(width - height) // 2
            max_c += abs(width - height) // 2
        else:
            min_c -= (abs(width - height) + 1) // 2
            max_c += (abs(width - height) - 1) // 2

    min_r = max(0, min_r)
    min_c = max(0, min_c)

    final_cropped = final[min_r:max_r, min_c:max_c]

    if only_cropped:
        final_cropped = transformed_image_array[min_r:max_r, min_c:max_c]

    return final_cropped
