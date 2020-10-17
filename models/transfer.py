import pathlib
from datetime import datetime

import numpy as np
import tensorflow as tf

from utils.logger import logger

BATCH_SIZE = 32
SEED = 123
IMAGE_SIZE = (299, 299)
IMAGE_SHAPE = IMAGE_SIZE + (3,)
TRAIN_EPOCH = 100
FINE_TUNE_EPOCH = 50
VALIDATION_SPLIT = 0.2

model_name = pathlib.Path(__file__).stem
model_timestamp = datetime.now().strftime("%m%d%H%M")

model_checkpoint_path = str(
    pathlib.Path(__file__).parent / f"{model_name}/{model_timestamp}"
)
model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=10, min_delta=0.001, mode="max"
)
logger.info(f"Model will be saved at {model_checkpoint_path}")

data_dir = str(
    (pathlib.Path.cwd().parent / "data/train_image/regrouped").resolve()
)
datagen_kwargs = dict(rescale=1.0 / 255, validation_split=VALIDATION_SPLIT)
dataflow_kwargs = dict(
    target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, interpolation="bilinear"
)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    zoom_range=0.3,
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    **datagen_kwargs,
)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs
)
train_generator = train_datagen.flow_from_directory(
    data_dir, subset="training", **dataflow_kwargs
)
valid_generator = validation_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=False, **dataflow_kwargs
)

base_model = tf.keras.applications.inception_v3.InceptionV3(
    input_shape=IMAGE_SHAPE, include_top=False, weights="imagenet"
)
output = base_model.layers[-1].output
base_model = tf.keras.Model(base_model.input, output)

base_model.trainable = False
base_model.summary()

model = tf.keras.Sequential(
    [
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(
            train_generator.num_classes,
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            activation="softmax",
        ),
    ]
)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)

history = model.fit(
    train_generator,
    epochs=TRAIN_EPOCH,
    validation_data=valid_generator,
    callbacks=[model_early_stopping_callback],
)

set_trainable = False
base_model.trainable = True
block_name = "block4"
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

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"{model_checkpoint_path}/fine-tuned-epoch-{{epoch:02d}}-acc-{{val_accuracy:.4f}}",
    verbose=0,
    save_weights_only=False,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)

history = model.fit(
    train_generator,
    epochs=TRAIN_EPOCH + FINE_TUNE_EPOCH,
    initial_epoch=history.epoch[-1],
    validation_data=valid_generator,
    callbacks=[model_checkpoint_callback, model_early_stopping_callback],
)
