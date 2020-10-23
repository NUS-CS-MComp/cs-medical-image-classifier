import pathlib
from datetime import datetime

import tensorflow as tf

from utils.configuration import MODEL_CHECKPOINT_PATH


def generate_model_checkpoint_callback(
    model_script_path: str, is_fine_tuning: bool = False
):
    """
    Helper function to generate checkpoint callback
    :param model_script_path: model script path passed from __file__
    :param is_fine_tuning: boolean flag for fine tuning
    :return: callback object
    """
    model_name = pathlib.Path(model_script_path).stem
    model_timestamp = datetime.now().strftime("%m%d%H%M")
    model_parent_path = (
        MODEL_CHECKPOINT_PATH / f"{model_name}/{model_timestamp}"
    )
    fine_tuning_flag = "fine-tuned-" if is_fine_tuning else ""
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{model_parent_path.absolute()}/{fine_tuning_flag}epoch-{{epoch:02d}}-acc-{{val_accuracy:.4f}}",
        verbose=0,
        save_weights_only=False,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    )


def generate_early_stopping_callback():
    """
    Helper function to general early stopping callback
    :return: callback object
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10, min_delta=0.001, mode="max"
    )
