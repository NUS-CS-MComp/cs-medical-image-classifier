import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf


def decode_image(image_file: object, image_height: int, image_width: int):
    image_data = tf.image.decode_png(image_file, channels=3)
    return tf.image.resize(image_data, [image_height, image_width])


def process_path(
    file_path: str, image_height: int = 224, image_width: int = 224
):
    image_file = tf.io.read_file(file_path)
    image_data = decode_image(
        image_file, image_height=image_height, image_width=image_width
    )
    # image_data /= 255.0
    return tf.expand_dims(image_data, 0)


image_path = tf.data.Dataset.list_files(
    str(
        pathlib.Path(__file__).parent.parent
        / "data/test_image/test_image"
        / "*.png"
    ),
    shuffle=False,
)
image_ds = image_path.map(
    process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE
)

model = tf.keras.models.load_model(
    "transfer/10172022/fine-tuned-epoch-29-acc-0.9307"
)
score = tf.nn.softmax(model.predict(image_ds))
prediction = np.argmax(score, axis=1)
assert prediction.size == len(image_path)

series = []
for path, prediction in zip(list(image_path.as_numpy_iterator()), prediction):
    series.append([pathlib.Path(path.decode("utf-8")).stem, prediction])

df = (
    pd.DataFrame(data=series, columns=["ID", "Label"])
    .astype(int)
    .sort_values("ID")
)
df.to_csv("prediction.csv", index=False)
