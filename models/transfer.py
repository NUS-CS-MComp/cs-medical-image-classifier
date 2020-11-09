from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from utils.logging import logger


def build_transfer_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    base_model: Model,
    global_average_pooling: bool = False,
) -> Tuple[Model, Model]:
    """
    Build transfer learning model
    :param input_shape: input data shape
    :param num_classes: Number of classes
    :param base_model: base model object
    :param global_average_pooling: boolean flag of using global average pooling
    :return: built model object
    """
    base_model = base_model(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    base_model.summary()

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    if global_average_pooling:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
    else:
        x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
    )(x)

    return tf.keras.Model(inputs, outputs), base_model


def unfreeze_layers(base_model: Model, from_block: str):
    """
    Unfreeze base model layers
    :param base_model: base model object
    :param from_block: block name as start of layer unfreezing
    :return: base model with unfrozen layers
    """
    set_trainable = False
    base_model.trainable = True
    for layer in base_model.layers:
        if from_block in layer.name:
            set_trainable = True
        if not set_trainable:
            layer.trainable = False
        if layer.trainable:
            logger.info(f"Unfreezing layer {layer.name}")

    sum_trainable_params = np.sum(
        [
            np.prod(v.get_shape().as_list())
            for v in base_model.trainable_variables
        ]
    )
    logger.info(f"Number of trainable params is now {sum_trainable_params}")
    return base_model


if __name__ == "__main__":
    from models.common.callbacks import (
        generate_early_stopping_callback,
        generate_model_checkpoint_callback,
        generate_learning_rate_schedule_callback,
    )
    from models.common.data_generator import DataGenerator
    from utils.configuration import (
        FINE_TUNE_EPOCH,
        TRAIN_DATA_DIR,
    )
    from utils.configuration import IMAGE_SHAPE, TRAIN_EPOCH

    model_tag = "vgg16" + (
        "_p" if "processed" in str(TRAIN_DATA_DIR) else "_o"
    )
    block_name_to_freeze = "block4"

    # model_state = dict(
    #     connection="ga",
    #     fc1=512,
    #     fc2=256,
    #     dropout=30,
    #     image_size=IMAGE_SHAPE[0],
    # )

    model_package = tf.keras.applications.vgg16
    original_model, preprocess_input = (
        model_package.VGG16,
        model_package.preprocess_input,
    )

    # from classification_models.tfkeras import Classifiers

    # original_model, preprocess_input = Classifiers.get("resnet18")

    # Data generator
    data_generator = DataGenerator(
        TRAIN_DATA_DIR, preprocessing_function=preprocess_input
    )
    training_dataset, validation_dataset = data_generator.data_generators

    # Callbacks
    model_early_stopping_callback = generate_early_stopping_callback()
    model_checkpoint_callback = generate_model_checkpoint_callback(
        __file__, tag=f"{model_tag}_{block_name_to_freeze}"
    )
    model_fine_tuning_checkpoint_callback = generate_model_checkpoint_callback(
        __file__,
        tag=f"{model_tag}_{block_name_to_freeze}",
        is_fine_tuning=True,
    )
    model_learning_rate_callback = generate_learning_rate_schedule_callback(
        start_learning_rate=1e-3, exp_decay=1e-2
    )
    model_fine_tuning_learning_rate_callback = (
        generate_learning_rate_schedule_callback(
            start_learning_rate=1e-5, exp_decay=1e-4
        )
    )

    # Training
    model, base = build_transfer_model(
        IMAGE_SHAPE,
        training_dataset.num_classes,
        original_model,
        global_average_pooling=True,
    )
    model.summary()

    # exit()

    # lr_schedule = tf.optimizers.schedules.ExponentialDecay(1e-3, 100, 0.9)
    # wd_schedule = tf.optimizers.schedules.ExponentialDecay(5e-5, 100, 0.9)
    # optimizer = tfa.optimizers.AdamW(
    #     learning_rate=lr_schedule, weight_decay=lambda: None
    # )
    # optimizer.weight_decay = lambda: wd_schedule(optimizer.iterations)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        training_dataset,
        epochs=TRAIN_EPOCH,
        validation_data=validation_dataset,
        callbacks=[
            model_learning_rate_callback,
            model_checkpoint_callback,
            model_early_stopping_callback,
        ],
    )

    # Fine-tuning
    base = unfreeze_layers(base, block_name_to_freeze)

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(
        training_dataset,
        epochs=TRAIN_EPOCH + FINE_TUNE_EPOCH,
        initial_epoch=history.epoch[-1],
        validation_data=validation_dataset,
        callbacks=[
            model_fine_tuning_learning_rate_callback,
            model_fine_tuning_checkpoint_callback,
            model_early_stopping_callback,
        ],
    )
