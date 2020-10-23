import tensorflow as tf

from models.common.callbacks import (
    generate_early_stopping_callback,
    generate_model_checkpoint_callback,
)
from models.common.data_generator import DataGenerator
from utils.configuration import BATCH_SIZE, TRAIN_DATA_DIR, TRAIN_EPOCH

model_early_stopping_callback = generate_early_stopping_callback()
model_checkpoint_callback = generate_model_checkpoint_callback(__file__)

data_generator = DataGenerator(TRAIN_DATA_DIR, rescale_data=True)
training_dataset, validation_dataset = data_generator.data_generators

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(training_dataset.num_classes),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=TRAIN_EPOCH,
    batch_size=BATCH_SIZE,
    callbacks=[model_early_stopping_callback, model_checkpoint_callback],
)
