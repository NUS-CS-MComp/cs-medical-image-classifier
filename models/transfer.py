import numpy as np
import tensorflow as tf

from models.common.callbacks import (
    generate_early_stopping_callback,
    generate_model_checkpoint_callback,
)
from models.common.data_generator import DataGenerator
from utils.configuration import (
    TRAIN_DATA_DIR,
    IMAGE_SHAPE,
    TRAIN_EPOCH,
    FINE_TUNE_EPOCH,
)
from utils.logging import logger

data_generator = DataGenerator(TRAIN_DATA_DIR, rescale_data=False)
training_dataset, validation_dataset = data_generator.data_generators

model_early_stopping_callback = generate_early_stopping_callback()
model_checkpoint_callback = generate_model_checkpoint_callback(__file__)

base_model = tf.keras.applications.vgg16.VGG16(
    input_shape=IMAGE_SHAPE, include_top=False, weights="imagenet"
)
base_model.trainable = False
base_model.summary()

prediction_layer = tf.keras.Sequential(
    [
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(
            training_dataset.num_classes,
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
        ),
    ]
)

inputs = tf.keras.Input(shape=IMAGE_SHAPE)
x = tf.keras.applications.vgg16.preprocess_input(inputs)
x = base_model(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(
    training_dataset,
    epochs=TRAIN_EPOCH,
    validation_data=validation_dataset,
    callbacks=[model_early_stopping_callback],
)

set_trainable = False
base_model.trainable = True
block_name = "block5"
for layer in base_model.layers:
    if block_name in layer.name:
        set_trainable = True
    if not set_trainable or "BatchNormalization" in str(layer.__class__):
        layer.trainable = False
    if layer.trainable:
        logger.info(f"Unfreezing layer {layer.name}")
sum_trainable_params = np.sum(
    [np.prod(v.get_shape().as_list()) for v in base_model.trainable_variables]
)
logger.info(f"Number of trainable params is now {sum_trainable_params}")

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)

history = model.fit(
    training_dataset,
    epochs=TRAIN_EPOCH + FINE_TUNE_EPOCH,
    initial_epoch=history.epoch[-1],
    validation_data=validation_dataset,
    callbacks=[model_checkpoint_callback, model_early_stopping_callback],
)
