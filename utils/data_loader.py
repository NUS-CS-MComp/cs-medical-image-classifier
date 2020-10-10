import pathlib
import re
from typing import Tuple

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from utils.logger import logger


class DataLoader:
    IMAGE_CACHE_NAME = "train_images.npy"
    LABEL_CACHE_NAME = "train_labels.npy"

    def __init__(
        self,
        image_path: pathlib.Path = pathlib.Path(__file__).parent
        / "../data/train_image/train_image",
        data_path: pathlib.Path = pathlib.Path(__file__).parent
        / "../data/train_label.csv",
        store_path: pathlib.Path = pathlib.Path(__file__).parent / "../data",
    ):
        self.data_path = data_path
        self.image_path = image_path
        self.store_path = store_path

        image_cache_path = store_path / DataLoader.IMAGE_CACHE_NAME
        label_cache_path = store_path / DataLoader.LABEL_CACHE_NAME
        self.images: np.ndarray = (
            np.array([])
            if not image_cache_path.exists()
            else np.load(str(image_cache_path))
        )
        self.labels: np.ndarray = (
            np.array([])
            if not label_cache_path.exists()
            else np.load(str(label_cache_path))
        )

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        def extract_id_from_filename(name: str) -> int:
            return int(re.findall(r"(\d+).png", name)[0])

        if self.images.size != 0 and self.labels.size != 0:
            logger.info(
                f"Using cached training data objects at {self.store_path.resolve().absolute()}"
            )
            return self.images, self.labels

        # Load images
        images = []
        image_ids = []
        logger.info(
            "Reading training data and images and converting them to numpy array"
        )

        for file in sorted(
            self.image_path.glob("*.png"),
            key=lambda x: extract_id_from_filename(x.name),
        ):
            image = mpimg.imread(file)
            image_ids.append(extract_id_from_filename(file.name))
            images.append(image)
        self.images = np.array(images)
        image_ids = np.array(image_ids)

        # Load labels
        label_df = pd.read_csv(self.data_path)
        ids = label_df.loc[:, "ID"].to_numpy()
        self.labels = label_df.loc[:, "Label"].to_numpy()

        # Consistency check
        assert np.sum(ids != image_ids) == 0
        assert ids.shape == image_ids.shape

        # Store in disk
        logger.info(
            f"Storing training data at {self.store_path.resolve().absolute()}"
        )
        np.save(
            str(self.store_path / DataLoader.IMAGE_CACHE_NAME), self.images
        )
        np.save(
            str(self.store_path / DataLoader.LABEL_CACHE_NAME), self.images
        )

        return self.images, self.labels


dl = DataLoader()
dl.load_data()
