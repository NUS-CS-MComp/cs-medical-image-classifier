import pathlib
from datetime import datetime

import tensorflow as tf

from utils.logger import logger

BATCH_SIZE = 32
IMAGE_SIZE = (229, 229)
IMAGE_SHAPE = IMAGE_SIZE + (3,)
TRAIN_EPOCH = 200
VALIDATION_SPLIT = 0.2

data_dir = str(
    (pathlib.Path.cwd().parent / "data/train_image/regrouped").resolve()
)
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=137,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=137,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)
class_names = train_dataset.class_names
num_classes = len(class_names)

model_name = pathlib.Path(__file__).stem
model_timestamp = datetime.now().strftime("%m%d%H%M")

model_checkpoint_path = str(
    pathlib.Path(__file__).parent / f"{model_name}/{model_timestamp}"
)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"{model_checkpoint_path}/epoch-{{epoch:02d}}-acc-{{val_accuracy:.4f}}",
    verbose=0,
    save_weights_only=False,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)
model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=8, min_delta=0.001, mode="max"
)
logger.info(f"Model will be saved at {model_checkpoint_path}")

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ]
)

model = tf.keras.models.Sequential(
    [
        augmentation,
        tf.keras.layers.experimental.preprocessing.Resizing(
            height=IMAGE_SIZE[0], width=IMAGE_SIZE[0]
        ),
        tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model.build((BATCH_SIZE,) + IMAGE_SHAPE)
model.summary()

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=TRAIN_EPOCH,
    batch_size=BATCH_SIZE,
    callbacks=[model_early_stopping_callback],
)

tf.saved_model.save(model, f"{model_checkpoint_path}/final")
