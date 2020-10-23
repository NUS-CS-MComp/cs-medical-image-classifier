import logging
import pathlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

from utils.configuration import (
    CONCURRENT_PROCESSING_THRESHOLD,
    TRAIN_LABEL_DATA_PATH,
    ORIGIN_TRAIN_DATA_DIR,
    ORIGIN_TEST_DATA_DIR,
    TRAIN_DATA_DIR,
    TEST_DATA_DIR,
)
from utils.image.segmentation import segment_from_image
from utils.logging import logger

LABEL_DF = pd.read_csv(TRAIN_LABEL_DATA_PATH).set_index("ID")


def process_image(
    image_path: pathlib.Path,
    target_dir: pathlib.Path,
    label_df: pd.DataFrame,
    infer_label: bool = True,
):
    """
    Process image given original image path
    :param image_path: image path object
    :param target_dir: target directory path object
    :param label_df: label dataframe
    :param infer_label: boolean flag to infer label from image
    :return: None
    """
    logger.setLevel(logging.DEBUG)

    label = (
        None
        if not infer_label
        else label_df.loc[int(image_path.stem), "Label"]
    )

    # Check if file already exists
    new_path = target_dir if label is None else target_dir / str(label)
    new_path.resolve().mkdir(parents=True, exist_ok=True)
    new_file = new_path / image_path.name
    if new_file.exists():
        logger.debug(f"File {new_file.resolve()} already exists")
        return

    # Process image if no file has been created
    image = Image.open(image_path)
    image_array = np.array(image)
    image_array_processed = segment_from_image(image_array)
    Image.fromarray(image_array_processed).save(new_file)
    logger.debug(f"Processed and added new file {new_file}")


if __name__ == "__main__":
    image_files = [
        file for file in ORIGIN_TRAIN_DATA_DIR.iterdir() if file.is_file()
    ]
    test_image_files = [
        file for file in ORIGIN_TEST_DATA_DIR.iterdir() if file.is_file()
    ]
    with Pool(CONCURRENT_PROCESSING_THRESHOLD) as p:
        p.starmap(
            process_image,
            (
                (
                    file,
                    TRAIN_DATA_DIR,
                    LABEL_DF,
                )
                for file in image_files
            ),
        )
        p.starmap(
            process_image,
            ((file, TEST_DATA_DIR, None, False) for file in test_image_files),
        )
