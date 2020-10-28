import tensorflow as tf

from models.common.callbacks import (
    generate_early_stopping_callback,
    generate_model_checkpoint_callback,
)
from models.common.data_generator import DataGenerator
from utils.configuration import (
    BATCH_SIZE,
    IMAGE_SHAPE,
    TRAIN_DATA_DIR,
    TRAIN_EPOCH,
)

model_early_stopping_callback = generate_early_stopping_callback()
model_checkpoint_callback = generate_model_checkpoint_callback(__file__)

data_generator = DataGenerator(TRAIN_DATA_DIR, None)
training_dataset, validation_dataset = data_generator.data_generators


def generate_vanilla_cnn_model():
    inputs = tf.keras.Input(shape=IMAGE_SHAPE)

    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(
        training_dataset.num_classes, activation="softmax"
    )(x)

    return tf.keras.Model(inputs, outputs)


model = generate_vanilla_cnn_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)

history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=TRAIN_EPOCH,
    batch_size=BATCH_SIZE,
    callbacks=[model_early_stopping_callback, model_checkpoint_callback],
)
