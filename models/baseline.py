import pathlib

import tensorflow as tf

BATCH_SIZE = 32
IMAGE_SIZE = (229, 229)
IMAGE_SHAPE = IMAGE_SIZE + (3,)
TRAIN_EPOCH = 30
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

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

train_dataset = train_dataset.prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE
)
validation_dataset = validation_dataset.prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE
)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        # tf.keras.layers.experimental.preprocessing.RandomHeight(0.2),
        # tf.keras.layers.experimental.preprocessing.RandomWidth(0.2),
        # tf.keras.layers.experimental.preprocessing.RandomTranslation(0.1, 0.1),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)

base_model = tf.keras.applications.ResNet50(
    input_shape=IMAGE_SHAPE, include_top=False, weights="imagenet"
)
base_model.summary()

preprocess = tf.keras.applications.resnet50.preprocess_input

# base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(
            len(class_names),
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        ),
    ]
)

inputs = tf.keras.Input(shape=IMAGE_SHAPE)
x = data_augmentation(inputs)
x = preprocess(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.001
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(
    train_dataset, epochs=TRAIN_EPOCH, validation_data=validation_dataset
)
