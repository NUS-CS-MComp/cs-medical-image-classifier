import os
import pathlib
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from utils.logger import logger


class DataLoader:
    """
    Generic image data loader with custom input workflow embedded in TensorFlow DataSet support for batching loading
    """

    def __init__(
        self,
        image_path: pathlib.Path = pathlib.Path(__file__).parent
        / "../data/train_image/train_image",
        data_path: pathlib.Path = pathlib.Path(__file__).parent
        / "../data/train_label.csv",
    ):
        """
        Initiate data loader
        :param image_path: Path instance for image path
        :param data_path: Path instance for data label file path
        """

        # Initiate path to read image and label files
        self.data_path = data_path
        self.image_path = image_path

        # Initiate labels, classes and training/validation set
        self.labels: Optional[pd.DataFrame] = None
        self.classes: Optional[np.ndarray] = None
        self.image_size: Optional[Tuple[int, int]] = None
        self.standardize_image = False

        self.training_ds: Optional[tf.data.Dataset] = None
        self.validation_ds: Optional[tf.data.Dataset] = None

    def load(
        self,
        batch_size=32,
        seed=123,
        image_height=512,
        image_width=512,
        validation_split=0.2,
        standardize=True,
    ):
        """
        Main function to load dataset object
        :param batch_size: Batch size
        :param seed: Seed for randomization
        :param image_height: Image height
        :param image_width: Image width
        :param validation_split: Train/validation split ratio
        :param standardize: Standardization flag
        :return: Pre-fetched training and validation TensorFlow dataset object
        """

        # Load label data
        logger.info(f"Reading label .csv data at {self.data_path.resolve()}")
        labels = pd.read_csv(self.data_path)
        self.labels = labels.set_index("ID")
        self.classes = sorted(
            labels["Label"].unique()
        )  # Important to sort class label in ascending order
        self.image_size = (image_height, image_width)
        self.standardize_image = standardize

        # Load image data
        logger.info(f"Reading image data at {self.image_path.resolve()}")
        image_ds = tf.data.Dataset.list_files(
            str(self.image_path / "*.png"), shuffle=True, seed=seed
        )
        image_count = len(image_ds)

        # Train-validation data split
        val_size = int(image_count * validation_split)
        train_ds = image_ds.skip(val_size)
        val_ds = image_ds.take(val_size)

        # Map custom process flow
        logger.info("Running custom input pipelines")
        train_ds = train_ds.map(
            self.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        val_ds = val_ds.map(
            self.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        # Add performance configuration on training and validation set
        logger.info("Tuning dataset batch loading performance")
        train_ds = DataLoader.configure_for_performance(
            train_ds, batch_size=batch_size
        )
        val_ds = DataLoader.configure_for_performance(
            val_ds, batch_size=batch_size
        )

        # Start dataset buffer loading and return prefetched instances
        self.training_ds = train_ds.cache().prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )
        self.validation_ds = val_ds.cache().prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )

        return self.training_ds, self.validation_ds

    def process_path(self, file_path: str):
        """
        Helper function to process dataset as customized workflow
        :param file_path: File path in string format
        :return: Parsed image data and corresponding label given the input file path
        """
        label_name = self.get_label(file_path)
        image_file = tf.io.read_file(file_path)
        image_data = DataLoader.decode_image(
            image_file,
            image_height=self.image_size[0],
            image_width=self.image_size[1],
            standardize=self.standardize_image,
        )
        return image_data, label_name

    def get_label(self, file_path: str):
        """
        Helper function to parse label class
        :param file_path: File path in string format
        :return: Single-sized label representation in Tensor form
        """
        file_name = tf.strings.split(file_path, os.sep)[-1]
        file_id = tf.strings.split(file_name, ".")[-2]
        label_name = tf.gather_nd(
            self.labels,
            [tf.strings.to_number(file_id, out_type=tf.dtypes.int32)],
        )
        return tf.argmax(label_name == self.classes)

    @property
    def training_data(self):
        return self.training_ds

    @property
    def validation_data(self):
        return self.validation_ds

    @staticmethod
    def configure_for_performance(dataset: tf.data.Dataset, batch_size: int):
        """
        Helper function from https://www.tensorflow.org/tutorials/load_data/images#configure_dataset_for_performance
        :param dataset: TensorFlow Dataset object
        :param batch_size: Batch size
        :return: Performance-wise tuned Dataset object
        """
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    @staticmethod
    def decode_image(
        image_file: object,
        image_height: int,
        image_width: int,
        standardize: bool,
    ):
        """
        Helper function to decode image file
        :param image_file: Image buffer file
        :param image_height: Image height
        :param image_width: Image width
        :param standardize: Standardization flag
        :return: Decoded image file in Tensor form
        """
        image_data = tf.image.decode_png(image_file, channels=3)
        if standardize:
            image_data = tf.image.per_image_standardization(image_data)
        return tf.image.resize(image_data, [image_height, image_width])
