import inspect
import pathlib
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from models.transfer import build_transfer_model
from utils.logging import logger


def decode_image(
    image_file: pathlib.Path, image_height: int, image_width: int
):
    """
    Read image path and decode as tensor
    :param image_file: image path object
    :param image_height: image height
    :param image_width: image width
    :return: resized image array
    """
    image_data = tf.image.decode_png(image_file, channels=3)
    return tf.image.resize(image_data, [image_height, image_width])


def process_image(
    image_height: int, image_width: int, preprocessing_function: Callable
):
    """
    Function to process image from file path
    :param image_height: image height
    :param image_width: image width
    :param preprocessing_function: preprocessing function
    :return: function thunk to pass to custom data pipeline
    """

    def process_path(file_path: str):
        """
        Thunk function to process image path
        :param file_path: file path in string format
        :return: processed image array
        """
        image_file = tf.io.read_file(file_path)
        image_data = decode_image(
            image_file, image_height=image_height, image_width=image_width
        )
        image_data = preprocessing_function(image_data)
        return tf.expand_dims(image_data, 0)

    return process_path


def generate_test_dataset(
    test_data_dir: pathlib.Path,
    preprocessing_function: Callable = lambda image: image / 255.0,
    image_height: int = 224,
    image_width: int = 224,
):
    """
    Generate testing dataset
    :param test_data_dir: test data directory path object
    :param preprocessing_function: preprocessing function default as rescaling to [0,1]
    :param image_height: image height
    :param image_width: image width
    :return: path and testing dataset object
    """
    image_path = tf.data.Dataset.list_files(
        str(test_data_dir / "*.png"), shuffle=False
    )
    return image_path, image_path.map(
        process_image(
            image_height=image_height,
            image_width=image_width,
            preprocessing_function=preprocessing_function,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


def generate_predictions(
    model_path: pathlib.Path,
    base_model: tf.keras.Model,
    image_path: tf.data.Dataset,
    testing_dataset: tf.data.Dataset,
    activation_function: str = "softmax",
):
    base_model.load_weights(model_path)
    activation_layer = dict(inspect.getmembers(tf.nn))[activation_function]
    score = activation_layer(base_model.predict(testing_dataset))
    prediction = np.argmax(score, axis=1)

    series = []
    for path, prediction in zip(
        list(image_path.as_numpy_iterator()), prediction
    ):
        series.append([pathlib.Path(path.decode("utf-8")).stem, prediction])

    predictions = (
        pd.DataFrame(data=series, columns=["ID", "Label"])
        .astype(int)
        .sort_values("ID")
    )
    predictions.to_csv(model_path.parent / "predictions.csv", index=False)
    logger.info(
        f"Predictions saved as {model_path.parent / 'predictions.csv'}"
    )
    return predictions


def load_transfer_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    base_model_name: str,
    global_average_pooling: bool,
):
    """
    Helper function to load transfer learning model
    :param input_shape: input shape tuple
    :param num_classes: number of classes
    :param base_model_name: base model name
    :param global_average_pooling: boolean flag of using global average pooling
    :return:
    """
    model_package = None
    base_model = None
    if base_model_name == "vgg16":
        model_package = tf.keras.applications.vgg16
        base_model = model_package.VGG16
    elif base_model_name == "vgg19":
        model_package = tf.keras.applications.vgg19
        base_model = model_package.VGG19
    elif base_model_name == "resnet50":
        model_package = tf.keras.applications.resnet50
        base_model = model_package.ResNet50
    elif base_model_name == "inception":
        model_package = tf.keras.applications.inception_v3
        base_model = model_package.InceptionV3
    else:
        raise ValueError("Model not supported")
    model, base_model = build_transfer_model(
        input_shape, num_classes, base_model, global_average_pooling
    )
    return model, model_package.preprocess_input


if __name__ == "__main__":
    from utils.configuration import (
        MODEL_CHECKPOINT_PATH,
        IMAGE_SHAPE,
        IMAGE_SIZE,
        TEST_DATA_DIR,
    )
    import tensorflow as tf

    activation = "softmax"
    path = (
        MODEL_CHECKPOINT_PATH
        / "transfer_vgg16_o_block4/10280104_cgavg_fc1512_fc2256_d30_i512/ft-epoch-53-loss-0.1087"
    )
    built_model, preprocessing = load_transfer_model(
        input_shape=IMAGE_SHAPE,
        num_classes=3,
        base_model_name="vgg16",
        global_average_pooling=True,
    )

    test_path, test_dataset = generate_test_dataset(
        test_data_dir=TEST_DATA_DIR,
        preprocessing_function=preprocessing,
        image_height=IMAGE_SIZE[0],
        image_width=IMAGE_SIZE[1],
    )

    generate_predictions(
        model_path=path,
        base_model=built_model,
        image_path=test_path,
        testing_dataset=test_dataset,
        activation_function=activation,
    )
