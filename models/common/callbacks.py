import pathlib
from datetime import datetime

import tensorflow as tf

from utils.configuration import MODEL_CHECKPOINT_PATH


def generate_model_checkpoint_callback(
    model_script_path: str, is_fine_tuning: bool = False, tag: str = ""
):
    """
    Helper function to generate checkpoint callback
    :param model_script_path: model script path passed from __file__
    :param is_fine_tuning: boolean flag for fine tuning
    :param tag: extra tag to append to model name
    :return: callback object
    """
    model_name = pathlib.Path(model_script_path).stem
    if tag != "":
        model_name = model_name + f"_{tag}"
    model_timestamp = datetime.now().strftime("%m%d%H%M")
    model_parent_path = (
        MODEL_CHECKPOINT_PATH / f"{model_name}/{model_timestamp}"
    )
    fine_tuning_flag = "fine-tuned-" if is_fine_tuning else ""
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{model_parent_path.absolute()}/{fine_tuning_flag}epoch-{{epoch:02d}}-loss-{{val_loss:.4f}}",
        verbose=1,
        save_weights_only=True,
        monitor="val_loss",
        save_best_only=True,
    )


def generate_early_stopping_callback():
    """
    Helper function to general early stopping callback
    :return: callback object
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, min_delta=0.001, mode="min"
    )
