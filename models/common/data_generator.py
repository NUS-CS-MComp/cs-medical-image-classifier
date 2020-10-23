import pathlib

import tensorflow as tf

from utils.configuration import VALIDATION_SPLIT, IMAGE_SIZE, BATCH_SIZE


class DataGenerator:
    """
    Data generator class
    """

    AUGMENTATION_OPTIONS = dict(
        zoom_range=0.1,
        rotation_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
    )
    DATAGEN_KWARGS = dict(validation_split=VALIDATION_SPLIT)
    DATAFLOW_KWARGS = dict(
        target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, interpolation="bilinear"
    )

    def __init__(self, data_dir: pathlib.Path, rescale_data: bool = True):
        """
        Initialize a training data generator
        :param data_dir: training data path object
        :param rescale_data: boolean flag to rescale data
        """
        self.data_dir = data_dir
        extra_config = dict(rescale=1.0 / 255.0) if rescale_data else {}
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            **DataGenerator.AUGMENTATION_OPTIONS,
            **DataGenerator.DATAGEN_KWARGS,
            **extra_config
        )
        validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            **DataGenerator.DATAGEN_KWARGS, **extra_config
        )
        self.training_data = train_datagen.flow_from_directory(
            data_dir, subset="training", **DataGenerator.DATAFLOW_KWARGS
        )
        self.validation_data = validation_datagen.flow_from_directory(
            data_dir, subset="validation", **DataGenerator.DATAFLOW_KWARGS
        )

    @property
    def data_generators(self):
        """
        Getter for training and validation dataset
        :return: training and validation data generator
        """
        return self.training_data, self.validation_data
