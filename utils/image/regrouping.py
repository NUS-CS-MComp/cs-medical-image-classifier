import logging
import pathlib
import shutil

import pandas as pd

from utils.logging import logger


def copy_image_to_label_folder(
    file: pathlib.Path, target_dir: pathlib.Path, label_df: pd.DataFrame
):
    """
    Copy image to provided label folder
    :param file: file path object
    :param target_dir: target directory path object
    :param label_df: label dataframe
    :return: target file path object
    """
    # Query by dataframe and create folder marked by label
    label = label_df.loc[int(file.stem), "Label"]
    new_path = target_dir / str(label)
    new_path.resolve().mkdir(parents=True, exist_ok=True)
    new_file = new_path / file.name
    if not new_file.exists():
        logger.debug(f"Copying file {file.name} as path {new_file.resolve()}")
        shutil.copyfile(file, new_file)
    else:
        logger.debug(
            f"Target file {file.name} already exists under {new_path.resolve()}"
        )
    return new_file


def regroup_data(
    label_file_path: pathlib.Path,
    origin_image_dir: pathlib.Path,
    target_image_dir: pathlib.Path,
):
    """
    Group images by labels and copy to separate paths
    :param label_file_path: label file path
    :param origin_image_dir: origin image directory
    :param target_image_dir: target path to copy grouped images
    :return: None
    """
    label_df = pd.read_csv(label_file_path).set_index("ID")
    for file in origin_image_dir.iterdir():
        if file.is_file():
            copy_image_to_label_folder(file, target_image_dir, label_df)


if __name__ == "__main__":
    from utils.configuration import (
        DATA_DIR,
        ORIGIN_TRAIN_DATA_DIR,
        TRAIN_LABEL_DATA_PATH,
    )

    TARGET_IMAGE_DIR = DATA_DIR / "train_image/regrouped"
    logger.setLevel(logging.DEBUG)
    regroup_data(
        label_file_path=TRAIN_LABEL_DATA_PATH,
        origin_image_dir=ORIGIN_TRAIN_DATA_DIR,
        target_image_dir=TARGET_IMAGE_DIR,
    )
